import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9568
prof = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots") / f"prof_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--window-size=1440,2600",
    "--disable-gpu",
    "--disable-extensions",
    f"--user-data-dir={prof}",
    "http://localhost:8501/?page=visual"
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3.0)

async def main():
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=3)
        targets = json.loads(resp.read().decode())
        ws_url = [t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page"][0]

        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            mid = 0
            async def send(m, p=None):
                nonlocal mid
                mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid:
                        return d

            await send("Page.enable")
            await send("Runtime.enable")
            await send("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 2600,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(4.0)

            # Click Landsat tab
            await send("Runtime.evaluate", {
                "expression": """(() => {
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const lTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                    if (lTab) lTab.click();
                })()"""
            })
            await asyncio.sleep(3.5)

            async def native_click(x, y):
                await send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": int(x), "y": int(y)})
                await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": int(x), "y": int(y), "button": "left", "clickCount": 1})
                await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": int(x), "y": int(y), "button": "left", "clickCount": 1})

            async def select_method(target_prefix):
                res = await send("Runtime.evaluate", {
                    "expression": """(() => {
                        const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                        const algoBox = sboxes.find(s => s.innerText.includes('Select Change Detection Algorithm'));
                        if (!algoBox) return null;
                        const btn = algoBox.querySelector('button[aria-label="Open"]');
                        btn.scrollIntoView({block: 'center'});
                        const r = btn.getBoundingClientRect();
                        return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                    })()""",
                    "returnByValue": True
                })
                btn_pos = res.get("result", {}).get("result", {}).get("value")
                await native_click(btn_pos["x"], btn_pos["y"])
                await asyncio.sleep(0.8)

                opt_res = await send("Runtime.evaluate", {
                    "expression": f"""(() => {{
                        const opts = Array.from(document.querySelectorAll('[role="option"]'));
                        const target = opts.find(o => o.innerText.includes('{target_prefix}'));
                        if (!target) return null;
                        const r = target.getBoundingClientRect();
                        return {{text: target.innerText, x: r.x + r.width / 2, y: r.y + r.height / 2}};
                    }})()""",
                    "returnByValue": True
                })
                opt_pos = opt_res.get("result", {}).get("result", {}).get("value")
                await native_click(opt_pos["x"], opt_pos["y"])
                await asyncio.sleep(3.0)

            # Function to extract all metrics
            async def get_all_metrics():
                js = """(() => {
                    const metrics = Array.from(document.querySelectorAll('[data-testid="stMetric"]')).map(m => {
                        const label = m.querySelector('[data-testid="stMetricLabel"]') ? m.querySelector('[data-testid="stMetricLabel"]').innerText : '';
                        const val = m.querySelector('[data-testid="stMetricValue"]') ? m.querySelector('[data-testid="stMetricValue"]').innerText : '';
                        return {label, val};
                    });
                    
                    // Confusion matrix text block
                    const matrixText = Array.from(document.querySelectorAll('div')).find(d => d.innerText && d.innerText.includes('Independent Confusion Matrix:'));
                    const matrixStr = matrixText ? matrixText.innerText : '';
                    
                    return {metrics, matrixStr};
                })()"""
                r = await send("Runtime.evaluate", {"expression": js, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value", {})

            methods = ["Method A", "Method B", "Method C", "Method D", "Method E"]
            metrics_summary = {}

            for m in methods:
                await select_method(m)
                data = await get_all_metrics()
                print(f"\n--- {m} Metrics ---")
                for met in data["metrics"]:
                    if any(k in met["label"] for k in ["Changed Pixels", "Surface Change", "Footprint", "Precision", "Recall", "Specificity", "IoU", "Dice"]):
                        print(f"  {met['label']}: {met['val']}")
                metrics_summary[m] = data

            # Save to JSON
            out_path = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\objective4_live_metrics_summary.json")
            out_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
            print(f"\nMetrics written to {out_path}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
