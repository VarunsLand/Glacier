"""Capture Great Aletsch and Unteraar benchmark screenshots."""
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

async def capture_bench(target_btn: str, filename: str):
    port = 9228
    prof = OUTPUT_DIR / f"prof_{target_btn}_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        f"--user-data-dir={prof}",
        "http://localhost:8501"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        page_ws = None
        for _ in range(30):
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1)
                page_ws = json.loads(resp.read().decode())[0]["webSocketDebuggerUrl"]
                if page_ws:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.2)
            
        async with websockets.connect(page_ws, max_size=30_000_000) as ws:
            mid = 0
            async def send_cmd(m, p=None):
                nonlocal mid
                mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid:
                        return d

            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 1400,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(4.0)

            # Navigate to retreat explorer
            js_nav = """
            (() => {
                const labels = Array.from(document.querySelectorAll('label[data-baseweb="radio"], div[role="radiogroup"] label'));
                for (let l of labels) {
                    if (l.innerText.includes("Glacier Retreat Explorer")) {
                        l.click();
                        return "Clicked retreat";
                    }
                }
                return "Not found";
            })()
            """
            r_nav = await send_cmd("Runtime.evaluate", {"expression": js_nav, "returnByValue": True})
            print("Nav result:", r_nav.get("result", {}).get("value"))
            await asyncio.sleep(4.0)

            # Click benchmark button
            js_btn = f"""
            (() => {{
                const btns = Array.from(document.querySelectorAll('button'));
                for (let b of btns) {{
                    if (b.textContent.includes("{target_btn}")) {{
                        b.click();
                        return "Clicked button: " + b.textContent;
                    }}
                }}
                return "Button not found among " + btns.map(b => b.textContent.trim()).join(", ");
            }})()
            """
            r_btn = await send_cmd("Runtime.evaluate", {"expression": js_btn, "returnByValue": True})
            print("Btn result:", r_btn.get("result", {}).get("value"))
            await asyncio.sleep(5.0)

            # Capture screenshot
            res = await send_cmd("Page.captureScreenshot", {"format": "png"})
            out_file = OUTPUT_DIR / filename
            out_file.write_bytes(base64.b64decode(res["result"]["data"]))
            print(f"Captured: {filename}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

async def main():
    await capture_bench("Great Aletsch", "after_retreat_aletsch_1440.png")
    await capture_bench("Unteraar", "after_retreat_unteraar_1440.png")

if __name__ == "__main__":
    asyncio.run(main())
