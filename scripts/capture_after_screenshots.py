"""Automated visual verification and multi-viewport screenshot capture."""
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

VIEWPORTS = [
    ("1440", 1440, 1400),
    ("1366", 1366, 1300),
    ("1024", 1024, 1300),
    ("768", 768, 1400),
    ("390", 390, 1600),
]

async def run_capture(page_type: str, benchmark_name: str, vp_label: str, width: int, height: int):
    port = 9222
    profile_dir = OUTPUT_DIR / f"chrome_prof_{port}_{width}_{page_type}_{benchmark_name}"
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
            raise RuntimeError("Could not connect to Chrome debugger")
            
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
            
            # Settle initial home load
            await asyncio.sleep(4.0)
            
            if page_type == "retreat":
                # Navigate to retreat explorer
                js_nav = """
                (() => {
                    const labels = Array.from(document.querySelectorAll('section[data-testid="stSidebar"] label[data-baseweb="radio"], section[data-testid="stSidebar"] div[role="radiogroup"] label'));
                    for (let l of labels) {
                        if (l.innerText.includes("Glacier Retreat Explorer")) {
                            l.click();
                            return "Clicked retreat";
                        }
                    }
                    return "Retreat label not found";
                })()
                """
                await send_cmd("Runtime.evaluate", {"expression": js_nav, "returnByValue": True})
                await asyncio.sleep(4.0)
                
                if benchmark_name != "rhone":
                    btn_text = "Great" if benchmark_name == "aletsch" else "Unteraar"
                    js_btn = f"""
                    (() => {{
                        const btns = Array.from(document.querySelectorAll('button'));
                        for (let b of btns) {{
                            if (b.innerText.includes("{btn_text}")) {{
                                b.click();
                                return "Clicked: " + b.innerText;
                            }}
                        }}
                        return "Button not found";
                    }})()
                    """
                    res = await send_cmd("Runtime.evaluate", {"expression": js_btn, "returnByValue": True})
                    print(f"[{benchmark_name}] {res.get('result', {}).get('value')}")
                    await asyncio.sleep(4.0)
            
            # Settle rendering
            await asyncio.sleep(1.5)
            
            # Capture screenshot
            out_filename = f"after_{page_type}_{benchmark_name}_{vp_label}.png" if page_type == "retreat" else f"after_home_{vp_label}.png"
            out_path = OUTPUT_DIR / out_filename
            res = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_data = res["result"]["data"]
            out_path.write_bytes(base64.b64decode(b64_data))
            print(f"Captured: {out_filename} ({width}x{height})")
            
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

async def main():
    print("--- 1. Home Dashboard across all viewports ---")
    for label, w, h in VIEWPORTS:
        await run_capture("home", "home", label, w, h)
        
    print("--- 2. Rhone Glacier across all viewports ---")
    for label, w, h in VIEWPORTS:
        await run_capture("retreat", "rhone", label, w, h)
        
    print("--- 3. Great Aletsch Glacier across all viewports ---")
    for label, w, h in VIEWPORTS:
        await run_capture("retreat", "aletsch", label, w, h)

    print("--- 4. Unteraar Glacier at 1440px ---")
    await run_capture("retreat", "unteraar", "1440", 1440, 1400)

if __name__ == "__main__":
    asyncio.run(main())
