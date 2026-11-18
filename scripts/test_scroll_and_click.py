import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9562
prof = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots") / f"prof_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--window-size=1440,2400",
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
                "height": 2400,
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
            await asyncio.sleep(3.0)

            # Scroll into view and get button coordinates
            rect_res = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                    const algoBox = sboxes.find(s => s.innerText.includes('Select Change Detection Algorithm'));
                    const btn = algoBox.querySelector('button[aria-label="Open"]');
                    btn.scrollIntoView({block: 'center'});
                    const r = btn.getBoundingClientRect();
                    return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                })()""",
                "returnByValue": True
            })
            coords = rect_res.get("result", {}).get("result", {}).get("value", {})
            print("Button coords after scroll:", coords)

            x = int(coords["x"])
            y = int(coords["y"])
            await send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
            await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
            await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
            await asyncio.sleep(1.0)

            # Check options
            find_opts = """(() => {
                const opts = Array.from(document.querySelectorAll('[role="option"]'));
                return opts.map(o => {
                    const r = o.getBoundingClientRect();
                    return {text: o.innerText, x: r.x + r.width / 2, y: r.y + r.height / 2};
                });
            })()"""
            r2 = await send("Runtime.evaluate", {"expression": find_opts, "returnByValue": True})
            opts = r2.get("result", {}).get("result", {}).get("value", [])
            print("Options after click:", json.dumps(opts, indent=2))

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
