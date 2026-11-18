"""Capture screenshots of GlacierVision at various viewports via Chrome DevTools Protocol."""
import subprocess
import time
import json
import base64
import urllib.request
import asyncio
from pathlib import Path
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def capture_app(action: str, output_path: Path, width: int = 1440, height: int = 1080):
    port = 9222
    profile_dir = OUTPUT_DIR / f"chrome_profile_{port}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
        "http://localhost:8501"
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
            raise RuntimeError("Failed to get page debugger WebSocket URL")
            
        async with websockets.connect(page_ws, max_size=20_000_000) as ws:
            msg_id = 0
            
            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method}
                if params:
                    payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("id") == msg_id:
                        return data

            # Override device metrics
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": width,
                "height": height,
                "deviceScaleFactor": 1,
                "mobile": width <= 768
            })
            
            # Wait for initial load
            await asyncio.sleep(4.0)
            
            if action == "home":
                pass
            elif action == "retreat_rhone":
                # Find all radio labels in sidebar
                js = """
                (() => {
                    const labels = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] label[data-baseweb="radio"], section[data-testid="stSidebar"] div[role="radiogroup"] label'));
                    for (let l of labels) {
                        if (l.innerText.includes("Glacier Retreat Explorer")) {
                            l.click();
                            return "Clicked: " + l.innerText;
                        }
                    }
                    return "Not found, total labels: " + labels.length;
                })()
                """
                eval_res = await send_cmd("Runtime.evaluate", {"expression": js, "returnByValue": True})
                print("JS Eval:", eval_res.get("result", {}).get("value"))
                await asyncio.sleep(5.0)
            elif action == "retreat_aletsch":
                js = """
                (() => {
                    const labels = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] label[data-baseweb="radio"], section[data-testid="stSidebar"] div[role="radiogroup"] label'));
                    for (let l of labels) {
                        if (l.innerText.includes("Glacier Retreat Explorer")) {
                            l.click();
                            return "Clicked: " + l.innerText;
                        }
                    }
                    return "Not found";
                })()
                """
                await send_cmd("Runtime.evaluate", {"expression": js, "returnByValue": True})
                await asyncio.sleep(4.0)
                # Select Great Aletsch
                js_select = """
                (() => {
                    const select = document.querySelector('div[data-baseweb="select"]');
                    if (select) {
                        select.click();
                        return "Clicked select";
                    }
                    return "Select not found";
                })()
                """
                await send_cmd("Runtime.evaluate", {"expression": js_select, "returnByValue": True})
                await asyncio.sleep(1.0)
                js_pick = """
                (() => {
                    const options = Array.from(document.querySelectorAll('li[role="option"]'));
                    for (let o of options) {
                        if (o.innerText.includes("Great Aletsch")) {
                            o.click();
                            return "Clicked option: " + o.innerText;
                        }
                    }
                    return "Option not found";
                })()
                """
                await send_cmd("Runtime.evaluate", {"expression": js_pick, "returnByValue": True})
                await asyncio.sleep(5.0)

            # Capture screenshot
            res = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_data = res["result"]["data"]
            output_path.write_bytes(base64.b64decode(b64_data))
            print(f"Captured: {output_path.name} ({width}x{height})")
            
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

async def main():
    await capture_app("retreat_rhone", OUTPUT_DIR / "before_retreat_rhone_1440.png", 1440, 1080)
    await capture_app("retreat_aletsch", OUTPUT_DIR / "before_retreat_aletsch_1440.png", 1440, 1080)

if __name__ == "__main__":
    asyncio.run(main())
