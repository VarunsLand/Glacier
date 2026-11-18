import asyncio
import json
import urllib.request
import websockets
import subprocess
import time
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
port = 9520
prof = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots\prof_landsat_check")
prof.mkdir(parents=True, exist_ok=True)
cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={port}",
    "--disable-gpu",
    f"--user-data-dir={prof}",
    "http://localhost:8501/?page=monitoring"
]
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2)

async def check():
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")
        targets = json.loads(resp.read().decode())
        ws_url = [t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page"][0]
        async with websockets.connect(ws_url, max_size=30_000_000) as ws:
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
                "height": 1200,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(3)

            click_tab2 = """(() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const landsatTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                if (landsatTab) { landsatTab.click(); return true; }
                return false;
            })()"""
            res = await send("Runtime.evaluate", {"expression": click_tab2})
            print("Tab clicked:", res)

            # Wait for content to stabilize
            await asyncio.sleep(4)

            body = await send("Runtime.evaluate", {"expression": "document.body.innerText"})
            text = body.get("result", {}).get("result", {}).get("value", "")
            
            phrases = [
                "1_10_1",
                "Total Valid Pixels",
                "65,536",
                "Detected Changed Pixels",
                "54,789",
                "Estimated Detected Pixel-Change Area",
                "49.310 km²",
                "Ground-Truth Reference Mask Validation",
                "Intersection over Union (IoU)",
                "Earlier observation",
                "Later observation",
                "Spectral difference",
                "Predicted deterministic change mask",
                "Combined prediction overlay"
            ]
            for phrase in phrases:
                print(f"Contains '{phrase}':", phrase in text)

            # Also capture full page screenshot
            import base64
            shot = await send("Page.captureScreenshot", {"format": "png"})
            data = shot.get("result", {}).get("data", "")
            if data:
                out_path = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots\landsat_bitemporal_6panel_verified.png")
                out_path.write_bytes(base64.b64decode(data))
                print("Screenshot saved to:", out_path)
    finally:
        proc.terminate()

asyncio.run(check())
