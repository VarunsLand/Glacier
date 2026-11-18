import asyncio
import json
import urllib.request
import websockets
import subprocess
import time
from pathlib import Path

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
port = 9522
prof = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots\prof_landsat_check_2")
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
                "height": 1400,
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
            await send("Runtime.evaluate", {"expression": click_tab2})
            await asyncio.sleep(4)

            body = await send("Runtime.evaluate", {"expression": "document.body.innerText"})
            text = body.get("result", {}).get("result", {}).get("value", "")
            
            # Print text segments around 65,536
            idx = text.find("65,536")
            if idx != -1:
                start = max(0, idx - 200)
                end = min(len(text), idx + 600)
                print("--- SEGMENT AROUND 65,536 ---")
                print(text[start:end])
                print("--- END SEGMENT ---")
            else:
                print("65,536 not found!")
    finally:
        proc.terminate()

asyncio.run(check())
