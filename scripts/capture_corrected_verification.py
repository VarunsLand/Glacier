"""Capture high-fidelity screenshots of all four benchmark glaciers and the Multi-CRS table."""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = [
    ("Rhone", "http://localhost:8501/?page=retreat&glacier=Rhone", "corrected_retreat_rhone_1440.png"),
    ("Great Aletsch", "http://localhost:8501/?page=retreat&glacier=Great+Aletsch", "corrected_retreat_aletsch_1440.png"),
    ("Unteraar", "http://localhost:8501/?page=retreat&glacier=Unteraar", "corrected_retreat_unteraar_1440.png"),
    ("Gorner", "http://localhost:8501/?page=retreat&glacier=Gorner", "corrected_retreat_gorner_1440.png"),
]

async def capture():
    port = 9330
    prof = OUTPUT_DIR / f"prof_bench_final_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "--disable-extensions",
        f"--user-data-dir={prof}",
        "about:blank"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Wait for debugger
        page_ws = None
        for _ in range(30):
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1)
                targets = json.loads(resp.read().decode())
                for t in targets:
                    if t.get("type") == "page" and "webSocketDebuggerUrl" in t:
                        page_ws = t["webSocketDebuggerUrl"]
                        break
                if page_ws:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)
            
        if not page_ws:
            raise RuntimeError(f"No page target found on port {port}")

        async with websockets.connect(page_ws, max_size=30_000_000) as ws:
            mid = 0
            async def send_cmd(m, p=None):
                nonlocal mid; mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid: return d

            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 1300,
                "deviceScaleFactor": 1,
                "mobile": False
            })

            for name, url, filename in BENCHMARKS:
                print(f"Navigating to {name}: {url}")
                await send_cmd("Page.navigate", {"url": url})
                
                # Wait until kpi-card is rendered
                card_rendered = False
                for _ in range(30):
                    await asyncio.sleep(0.5)
                    eval_res = await send_cmd("Runtime.evaluate", {
                        "expression": "document.querySelectorAll('.kpi-card').length"
                    })
                    val = eval_res.get("result", {}).get("result", {}).get("value", 0)
                    if val >= 4:
                        card_rendered = True
                        break
                
                # Settle time
                await asyncio.sleep(2.0)
                
                # Capture viewport
                res = await send_cmd("Page.captureScreenshot", {"format": "png"})
                out_file = OUTPUT_DIR / filename
                out_file.write_bytes(base64.b64decode(res["result"]["data"]))
                print(f"Successfully captured: {filename} ({len(res['result']['data'])} bytes)")

                # If Rhone, let's open the expander for Multi-CRS table and capture it
                if name == "Rhone":
                    print("Opening Multi-CRS table expander...")
                    await send_cmd("Runtime.evaluate", {
                        "expression": """
                        (() => {
                            const details = Array.from(document.querySelectorAll('details'));
                            const crsExpander = details.find(d => d.textContent.includes('Spatial Projection'));
                            if (crsExpander) {
                                crsExpander.open = true;
                                crsExpander.scrollIntoView({ behavior: 'instant', block: 'center' });
                                return true;
                            }
                            return false;
                        })()
                        """
                    })
                    await asyncio.sleep(1.5)
                    res_table = await send_cmd("Page.captureScreenshot", {"format": "png"})
                    out_table = OUTPUT_DIR / "corrected_retreat_rhone_crs_table_1440.png"
                    out_table.write_bytes(base64.b64decode(res_table["result"]["data"]))
                    print("Successfully captured Multi-CRS table: corrected_retreat_rhone_crs_table_1440.png")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(capture())
