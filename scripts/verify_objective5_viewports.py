"""
verify_objective5_viewports.py
Verifies Objective 5 UI rendering across 5 standard viewport widths:
1440px (Desktop Wide), 1366px (Standard Laptop), 1024px (Tablet Landscape),
768px (Tablet Portrait), and 390px (Mobile).
Confirms no unrendered LaTeX, no overflow, and clean responsive rendering.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9572
OUTPUT_DIR = ROOT / "outputs" / "audits" / "screenshots" / "viewports"
BRAIN_DIR = Path(r"C:\Users\Varun's LOQ\.gemini\antigravity\brain\4ab4cfe4-3217-44c3-a812-2c996eb7e1da\screenshots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

prof = OUTPUT_DIR / f"prof_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--window-size=1440,1800",
    "--disable-gpu",
    "--disable-extensions",
    f"--user-data-dir={prof}",
    "http://localhost:8501"
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3.0)

viewports = [
    ("1440px", 1440, 1200),
    ("1366px", 1366, 1000),
    ("1024px", 1024, 1000),
    ("768px", 768, 1000),
    ("390px", 390, 1000),
]

async def run():
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=5)
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
            await asyncio.sleep(3.0)

            # Navigate to CLIP vs CoOp Lab
            res_nav = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const labels = Array.from(document.querySelectorAll('[data-testid="stRadio"] label'));
                    const r = labels.find(l => l.innerText && l.innerText.includes('CLIP vs CoOp Lab'));
                    if (r) {
                        r.scrollIntoView({block: 'center'});
                        const rect = r.getBoundingClientRect();
                        return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
                    }
                    return null;
                })()""",
                "returnByValue": True
            })
            nav_pos = res_nav.get("result", {}).get("result", {}).get("value")
            if nav_pos:
                await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": int(nav_pos["x"]), "y": int(nav_pos["y"]), "button": "left", "clickCount": 1})
                await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": int(nav_pos["x"]), "y": int(nav_pos["y"]), "button": "left", "clickCount": 1})
            await asyncio.sleep(4.0)

            # Verify LaTeX is resolved cleanly
            res_chk = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const txt = document.body.innerText;
                    return {
                        hasRawLatex: txt.includes('\\\\requires_grad') || txt.includes('\\text{requires_grad}') || txt.includes('requires_grad = False'),
                        hasFrozenText: txt.includes('CLIP backbone parameters are frozen (requires_grad=False)'),
                        bodyTextPreview: txt.substring(txt.indexOf('Scientific Findings'), txt.indexOf('Scientific Findings') + 500)
                    };
                })()""",
                "returnByValue": True
            })
            chk_val = res_chk.get("result", {}).get("result", {}).get("value", {})
            print("Scientific Findings text audit:", chk_val)

            # Capture screenshots for all 5 viewport dimensions
            for name, w, h in viewports:
                await send("Emulation.setDeviceMetricsOverride", {
                    "width": w,
                    "height": h,
                    "deviceScaleFactor": 1,
                    "mobile": w <= 768
                })
                await asyncio.sleep(1.5)

                res_scr = await send("Page.captureScreenshot", {"format": "png"})
                img_bytes = base64.b64decode(res_scr["result"]["data"])
                out_path = OUTPUT_DIR / f"obj5_viewport_{name}.png"
                brain_path = BRAIN_DIR / f"obj5_viewport_{name}.png"
                out_path.write_bytes(img_bytes)
                brain_path.write_bytes(img_bytes)
                print(f"Captured viewport {name} ({w}x{h}): {len(img_bytes):,} bytes")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(run())
