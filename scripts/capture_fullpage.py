"""Capture high-definition full-page screenshots of GlacierVision pages."""
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

async def capture_page(target_page: str, benchmark_btn: str, filename: str, width: int = 1440):
    port = 9226
    prof = OUTPUT_DIR / f"prof_full_{port}_{width}"
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
                "width": width,
                "height": 1080,
                "deviceScaleFactor": 1,
                "mobile": width <= 768
            })
            await asyncio.sleep(4.0)

            if target_page == "retreat":
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
                await send_cmd("Runtime.evaluate", {"expression": js_nav, "returnByValue": True})
                await asyncio.sleep(4.0)

                if benchmark_btn:
                    js_btn = f"""
                    (() => {{
                        const btns = Array.from(document.querySelectorAll('button'));
                        for (let b of btns) {{
                            if (b.innerText.includes("{benchmark_btn}")) {{
                                b.click();
                                return "Clicked: " + b.innerText;
                            }}
                        }}
                        return "Button not found";
                    }})()
                    """
                    await send_cmd("Runtime.evaluate", {"expression": js_btn, "returnByValue": True})
                    await asyncio.sleep(4.0)

            # Settle rendering
            await asyncio.sleep(1.5)

            # Capture full page beyond viewport
            res = await send_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
            out_file = OUTPUT_DIR / filename
            out_file.write_bytes(base64.b64decode(res["result"]["data"]))
            print(f"Captured full page: {filename}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

async def main():
    await capture_page("home", "", "after_home_full_1440.png", 1440)
    await capture_page("retreat", "", "after_retreat_rhone_full_1440.png", 1440)
    await capture_page("retreat", "Great", "after_retreat_aletsch_full_1440.png", 1440)
    await capture_page("retreat", "Unteraar", "after_retreat_unteraar_full_1440.png", 1440)

if __name__ == "__main__":
    asyncio.run(main())
