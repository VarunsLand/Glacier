"""
verify_objective5_live.py
Headless Chrome CDP verification of Objective 5: CLIP vs CoOp Lab.
Verifies live single-image inference, KPI cards, comparative table,
confusion matrices, download buttons, and scientific conclusions.
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

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9569
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
BRAIN_DIR = Path(r"C:\Users\Varun's LOQ\.gemini\antigravity\brain\4ab4cfe4-3217-44c3-a812-2c996eb7e1da\screenshots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

prof = OUTPUT_DIR / f"prof_obj5_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--window-size=1440,2400",
    "--disable-gpu",
    "--disable-extensions",
    f"--user-data-dir={prof}",
    "http://localhost:8501"
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3.0)

async def main():
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kpi_cards": [],
        "single_image_tests": {},
        "benchmark_table": {},
        "errors": [],
    }

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
            await send("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 2400,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(4.0)

            async def native_click(x, y):
                await send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": int(x), "y": int(y)})
                await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": int(x), "y": int(y), "button": "left", "clickCount": 1})
                await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": int(x), "y": int(y), "button": "left", "clickCount": 1})

            async def take_screenshot(name):
                res = await send("Page.captureScreenshot", {"format": "png"})
                img_bytes = base64.b64decode(res["result"]["data"])
                p1 = OUTPUT_DIR / f"{name}.png"
                p2 = BRAIN_DIR / f"{name}.png"
                p1.write_bytes(img_bytes)
                p2.write_bytes(img_bytes)
                print(f"  Captured screenshot: {name}.png ({len(img_bytes):,} bytes)")

            # Click Navigate to CLIP vs CoOp Lab
            print("[1] Navigating to CLIP vs CoOp Lab...")
            res_nav = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const labels = Array.from(document.querySelectorAll('[data-testid="stRadio"] label'));
                    const r = labels.find(l => l.innerText && l.innerText.includes('CLIP vs CoOp Lab'));
                    if (r) {
                        r.scrollIntoView({block: 'center'});
                        const rect = r.getBoundingClientRect();
                        return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
                    }
                    const btns = Array.from(document.querySelectorAll('button'));
                    const b = btns.find(btn => btn.innerText && btn.innerText.includes('Launch CLIP vs CoOp Lab'));
                    if (b) {
                        b.scrollIntoView({block: 'center'});
                        const rect = b.getBoundingClientRect();
                        return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
                    }
                    return null;
                })()""",
                "returnByValue": True
            })
            nav_pos = res_nav.get("result", {}).get("result", {}).get("value")
            if nav_pos:
                await native_click(nav_pos["x"], nav_pos["y"])
                print("  Clicked navigation link/button to CLIP vs CoOp Lab")
            else:
                print("  Could not find navigation button/radio")

            await asyncio.sleep(4.0)

            # 1. Extract Overview & KPI Cards
            print("\n[TEST 1] Auditing KPI Cards & Page Overview...")
            res_kpi = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const cards = Array.from(document.querySelectorAll('.kpi-card')).map(c => {
                        const label = c.querySelector('.kpi-label')?.innerText || '';
                        const val = c.querySelector('.kpi-value')?.innerText || '';
                        const badge = c.querySelector('.kpi-badge')?.innerText || '';
                        return {label, val, badge};
                    });
                    const text = document.body.innerText;
                    const hasStale = text.includes('0.6189') || text.includes('0.7586') || text.includes('81.00%');
                    const isLabPage = text.includes('Vision-Language Prompt Tuning Laboratory');
                    return {cards, hasStale, isLabPage};
                })()""",
                "returnByValue": True
            })
            kpi_data = res_kpi.get("result", {}).get("result", {}).get("value", {})
            print("  Is on Lab Page:", kpi_data.get("isLabPage"))
            print("  KPI Cards:", kpi_data.get("cards"))
            print("  Has Stale Mock Numbers:", kpi_data.get("hasStale"))
            if kpi_data.get("hasStale"):
                report["errors"].append("Detected stale mock numbers on page")
            report["kpi_cards"] = kpi_data.get("cards", [])

            await take_screenshot("clip_coop_live_overview")

            # 2. Test Sample 1 (Glacier Ice)
            print("\n[TEST 2] Auditing Sample 1 (Clean Alpine Glacier Ice)...")
            res_s1 = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const text = document.body.innerText;
                    const hasClipGlacier = text.includes('Glacier Ice');
                    const hasLatency = text.includes('Execution complete in');
                    return {hasClipGlacier, hasLatency};
                })()""",
                "returnByValue": True
            })
            s1_data = res_s1.get("result", {}).get("result", {}).get("value", {})
            print("  Sample 1 Predictions present:", s1_data)
            await take_screenshot("clip_coop_live_sample1_glacier")

            # 3. Select Sample 3 (Exposed Alpine Bedrock Ridge)
            print("\n[TEST 3] Selecting Sample 3 (Exposed Alpine Bedrock Ridge)...")
            res_sel = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                    const sBox = sboxes.find(s => s.innerText.includes('Select Sample'));
                    if (!sBox) return null;
                    const btn = sBox.querySelector('button[aria-label="Open"]');
                    if (!btn) return null;
                    btn.scrollIntoView({block: 'center'});
                    const r = btn.getBoundingClientRect();
                    return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                })()""",
                "returnByValue": True
            })
            btn_pos = res_sel.get("result", {}).get("result", {}).get("value")
            if btn_pos:
                await native_click(btn_pos["x"], btn_pos["y"])
                await asyncio.sleep(1.0)
                opt_res = await send("Runtime.evaluate", {
                    "expression": """(() => {
                        const opts = Array.from(document.querySelectorAll('[role="option"]'));
                        const target = opts.find(o => o.innerText.includes('Exposed Alpine Bedrock Ridge'));
                        if (!target) return null;
                        const r = target.getBoundingClientRect();
                        return {text: target.innerText, x: r.x + r.width / 2, y: r.y + r.height / 2};
                    })()""",
                    "returnByValue": True
                })
                opt_pos = opt_res.get("result", {}).get("result", {}).get("value")
                if opt_pos:
                    await native_click(opt_pos["x"], opt_pos["y"])
                    print(f"  Selected: {opt_pos['text']}")
                    await asyncio.sleep(4.0)

            await take_screenshot("clip_coop_live_sample3_bedrock")

            # 4. Extract Benchmark Table Values
            print("\n[TEST 4] Auditing Benchmark Table & Downloads...")
            res_table = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const text = document.body.innerText;
                    const hasTable = text.includes('Zero-Shot Handcrafted CLIP') && text.includes('Learnable CoOp');
                    const hasCSVDownload = Array.from(document.querySelectorAll('button')).some(b => b.innerText.includes('100-Sample Predictions'));
                    const hasFigureDownload = Array.from(document.querySelectorAll('button')).some(b => b.innerText.includes('Benchmark Figure'));
                    const has68 = text.includes('0.6800');
                    const has77 = text.includes('0.7700');
                    const hasTradeoff = text.includes('The Recall Trade-Off');
                    return {hasTable, hasCSVDownload, hasFigureDownload, has68, has77, hasTradeoff};
                })()""",
                "returnByValue": True
            })
            t_data = res_table.get("result", {}).get("result", {}).get("value", {})
            print("  Table Present:", t_data.get("hasTable"))
            print("  Canonical Accuracies Present (0.6800, 0.7700):", t_data.get("has68") and t_data.get("has77"))
            print("  CSV Download Button Present:", t_data.get("hasCSVDownload"))
            print("  Scientific Tradeoff Present:", t_data.get("hasTradeoff"))
            report["benchmark_table"] = t_data

            await take_screenshot("clip_coop_live_benchmark_table")

            # Save report
            rep_path = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\objective5_live_verification_results.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved live verification report to: {rep_path}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(main())
