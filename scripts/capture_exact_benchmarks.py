"""Capture all 4 benchmark glacier pages using clean query parameters."""
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
    ("rhone", "http://localhost:8501/?page=retreat&glacier=Rhone", "corrected_retreat_rhone_1440.png"),
    ("aletsch", "http://localhost:8501/?page=retreat&glacier=Great+Aletsch", "corrected_retreat_aletsch_1440.png"),
    ("unteraar", "http://localhost:8501/?page=retreat&glacier=Unteraar", "corrected_retreat_unteraar_1440.png"),
    ("gorner", "http://localhost:8501/?page=retreat&glacier=Gorner", "corrected_retreat_gorner_1440.png"),
]

async def capture_url(url: str, filename: str, port: int):
    prof = OUTPUT_DIR / f"prof_bench_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "--disable-extensions",
        f"--user-data-dir={prof}",
        url
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
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
                "height": 1400,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            # Wait for Streamlit React rendering to complete
            await asyncio.sleep(5.0)

            res = await send_cmd("Page.captureScreenshot", {"format": "png"})
            out_file = OUTPUT_DIR / filename
            out_file.write_bytes(base64.b64decode(res["result"]["data"]))
            print(f"Captured: {filename} from {url}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

async def main():
    p = 9250
    for name, url, fname in BENCHMARKS:
        await capture_url(url, fname, p)
        p += 1

if __name__ == "__main__":
    asyncio.run(main())
