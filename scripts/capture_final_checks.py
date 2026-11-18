"""Capture Home Dashboard and Objective 2 at both Desktop (1440px) and Mobile (390px)."""
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

TARGETS = [
    # Desktop 1440px
    {"name": "Home Dashboard (1440px)", "url": "http://localhost:8501/?page=home", "file": "final_home_1440.png", "width": 1440, "height": 1100, "mobile": False},
    {"name": "Unteraar Retreat Explorer (1440px)", "url": "http://localhost:8501/?page=retreat&glacier=Unteraar", "file": "final_retreat_unteraar_1440.png", "width": 1440, "height": 1300, "mobile": False},
    # Mobile 390px
    {"name": "Home Dashboard (390px)", "url": "http://localhost:8501/?page=home", "file": "final_home_390.png", "width": 390, "height": 1600, "mobile": True},
    {"name": "Rhone Retreat Explorer (390px)", "url": "http://localhost:8501/?page=retreat&glacier=Rhone", "file": "final_retreat_rhone_390.png", "width": 390, "height": 1800, "mobile": True},
    {"name": "Unteraar Retreat Explorer (390px)", "url": "http://localhost:8501/?page=retreat&glacier=Unteraar", "file": "final_retreat_unteraar_390.png", "width": 390, "height": 1800, "mobile": True},
]

async def capture_all():
    port = 9410
    prof = OUTPUT_DIR / f"prof_final_checks_{port}"
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
            raise RuntimeError(f"No page debugger found on port {port}")

        async with websockets.connect(page_ws, max_size=30_000_000) as ws:
            mid = 0
            async def send_cmd(m, p=None):
                nonlocal mid; mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid: return d

            for item in TARGETS:
                print(f"Setting viewport {item['width']}x{item['height']} (mobile={item['mobile']}) for {item['name']}...")
                await send_cmd("Emulation.setDeviceMetricsOverride", {
                    "width": item["width"],
                    "height": item["height"],
                    "deviceScaleFactor": 1,
                    "mobile": item["mobile"]
                })
                print(f"Navigating to {item['url']}...")
                await send_cmd("Page.navigate", {"url": item["url"]})

                # Wait for .kpi-card to render
                for _ in range(30):
                    await asyncio.sleep(0.5)
                    eval_res = await send_cmd("Runtime.evaluate", {
                        "expression": "document.querySelectorAll('.kpi-card').length"
                    })
                    val = eval_res.get("result", {}).get("result", {}).get("value", 0)
                    if val >= 4:
                        break

                # If mobile, collapse sidebar if open so main content has full width
                if item["mobile"]:
                    await send_cmd("Runtime.evaluate", {
                        "expression": """
                        (() => {
                            const sb = document.querySelector('section[data-testid="stSidebar"]');
                            if (sb) {
                                // Try finding collapse button
                                const closeBtn = document.querySelector('button[aria-label="Close sidebar"], button[data-testid="stSidebarCollapseButton"], [data-testid="stSidebar"] button');
                                if (closeBtn) closeBtn.click();
                            }
                        })()
                        """
                    })
                    await asyncio.sleep(1.0)

                await asyncio.sleep(1.5)

                # Check horizontal scroll/overflow
                overflow_res = await send_cmd("Runtime.evaluate", {
                    "expression": "document.documentElement.scrollWidth > document.documentElement.clientWidth"
                })
                has_overflow = overflow_res.get("result", {}).get("result", {}).get("value", False)
                print(f"  Horizontal overflow detected: {has_overflow}")

                res = await send_cmd("Page.captureScreenshot", {"format": "png"})
                out_path = OUTPUT_DIR / item["file"]
                out_path.write_bytes(base64.b64decode(res["result"]["data"]))
                print(f"  Captured: {item['file']} ({len(res['result']['data'])} bytes)")

                # If mobile retreat, also capture scrolled view for KPI cards
                if item["mobile"] and "retreat" in item["url"]:
                    await send_cmd("Runtime.evaluate", {
                        "expression": """
                        (() => {
                            const kpiGrid = document.querySelector('.kpi-grid-4, .kpi-card');
                            if (kpiGrid) kpiGrid.scrollIntoView({ behavior: 'instant', block: 'start' });
                        })()
                        """
                    })
                    await asyncio.sleep(1.0)
                    res_kpi = await send_cmd("Page.captureScreenshot", {"format": "png"})
                    kpi_file = OUTPUT_DIR / f"{item['file'].replace('.png', '_kpis.png')}"
                    kpi_file.write_bytes(base64.b64decode(res_kpi["result"]["data"]))
                    print(f"  Captured scrolled KPI view: {kpi_file.name}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(capture_all())
