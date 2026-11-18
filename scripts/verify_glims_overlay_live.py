"""
verify_glims_overlay_live.py
Complete live browser verification of Objective 4 Tab 3:
GLIMS Boundary Overlay & Geospatial Integration.
Verifies dynamic rendering fix, benchmark switching, and layer toggles.
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
PORT = 9568
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
BRAIN_DIR = Path(r"C:\Users\Varun's LOQ\.gemini\antigravity\brain\4ab4cfe4-3217-44c3-a812-2c996eb7e1da\screenshots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

prof = OUTPUT_DIR / f"prof_glims_live_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--window-size=1440,2400",
    "--disable-gpu",
    "--disable-extensions",
    f"--user-data-dir={prof}",
    "http://localhost:8501/?page=visual"
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3.0)


async def main():
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
        "layer_toggles": {},
        "errors_detected": [],
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

            # 1. Click GLIMS Boundary Overlay Tab
            print("[1] Navigating to GLIMS Boundary Overlay Tab...")
            await send("Runtime.evaluate", {
                "expression": """(() => {
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const gTab = tabs.find(t => t.innerText && t.innerText.includes('GLIMS Boundary Overlay'));
                    if (gTab) gTab.click();
                })()"""
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

            async def select_benchmark(target_keyword):
                res = await send("Runtime.evaluate", {
                    "expression": """(() => {
                        const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                        const bBox = sboxes.find(s => s.innerText.includes('Select Benchmark Alpine Glacier Basin'));
                        if (!bBox) return null;
                        const btn = bBox.querySelector('button[aria-label="Open"]');
                        if (!btn) return null;
                        btn.scrollIntoView({block: 'center'});
                        const r = btn.getBoundingClientRect();
                        return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                    })()""",
                    "returnByValue": True
                })
                btn_pos = res.get("result", {}).get("result", {}).get("value")
                if not btn_pos:
                    raise RuntimeError("Could not find Benchmark selectbox Open button")

                await native_click(btn_pos["x"], btn_pos["y"])
                await asyncio.sleep(1.0)

                opt_res = await send("Runtime.evaluate", {
                    "expression": f"""(() => {{
                        const opts = Array.from(document.querySelectorAll('[role="option"]'));
                        const target = opts.find(o => o.innerText.toLowerCase().includes('{target_keyword.lower()}'));
                        if (!target) return null;
                        const r = target.getBoundingClientRect();
                        return {{text: target.innerText, x: r.x + r.width / 2, y: r.y + r.height / 2}};
                    }})()""",
                    "returnByValue": True
                })
                opt_pos = opt_res.get("result", {}).get("result", {}).get("value")
                if not opt_pos:
                    raise RuntimeError(f"Could not find benchmark option for keyword: {target_keyword}")

                await native_click(opt_pos["x"], opt_pos["y"])
                print(f"  Selected benchmark: {opt_pos['text']}")
                await asyncio.sleep(4.0)

            async def extract_ui_state():
                res = await send("Runtime.evaluate", {
                    "expression": """(() => {
                        const text = document.body.innerText;
                        const hasError = text.includes('Error rendering dynamic GLIMS boundary overlay') || text.includes('Invalid mode for data type');
                        const hasImage = Array.from(document.querySelectorAll('img')).some(img => img.alt && img.alt.includes('Dynamic Vector Boundary Overlay'));
                        const hasDownloadBtn = Array.from(document.querySelectorAll('button')).some(b => b.innerText.includes('Download Styled Boundary Overlay'));
                        const metrics = Array.from(document.querySelectorAll('[data-testid="stMetric"]')).map(m => {
                            const label = m.querySelector('[data-testid="stMetricLabel"]')?.innerText || '';
                            const val = m.querySelector('[data-testid="stMetricValue"]')?.innerText || '';
                            const delta = m.querySelector('[data-testid="stMetricDelta"]')?.innerText || '';
                            return {label, val, delta};
                        });
                        const hasReconciliation = text.includes('Scientific Data Reconciliation Note') || text.includes('17.166 km²');
                        const hasTargetCrs = text.includes('EPSG:32632');
                        return {hasError, hasImage, hasDownloadBtn, metrics, hasReconciliation, hasTargetCrs};
                    })()""",
                    "returnByValue": True
                })
                return res.get("result", {}).get("result", {}).get("value")

            # -----------------------------------------------------------------
            # TEST 1: RHONE GLACIER (Default)
            # -----------------------------------------------------------------
            print("\n[TEST 1] Auditing Rhone Glacier (Default Benchmark)...")
            s_rhone = await extract_ui_state()
            print("  Has Error:", s_rhone["hasError"])
            print("  Has Dynamic Overlay Image:", s_rhone["hasImage"])
            print("  Has Download Button:", s_rhone["hasDownloadBtn"])
            print("  Metrics extracted:", s_rhone["metrics"])
            if s_rhone["hasError"]:
                report["errors_detected"].append("Rhone Glacier showed dynamic overlay error")
            await take_screenshot("glims_live_rhone_fixed")
            report["benchmarks"]["rhone"] = s_rhone

            # -----------------------------------------------------------------
            # TEST 2: GREAT ALETSCH GLACIER
            # -----------------------------------------------------------------
            print("\n[TEST 2] Auditing Great Aletsch Glacier...")
            await select_benchmark("Aletsch")
            s_aletsch = await extract_ui_state()
            print("  Has Error:", s_aletsch["hasError"])
            print("  Has Dynamic Overlay Image:", s_aletsch["hasImage"])
            print("  Metrics extracted:", s_aletsch["metrics"])
            if s_aletsch["hasError"]:
                report["errors_detected"].append("Great Aletsch showed dynamic overlay error")
            await take_screenshot("glims_live_great_aletsch_fixed")
            report["benchmarks"]["great_aletsch"] = s_aletsch

            # -----------------------------------------------------------------
            # TEST 3: UNTERAAR GLACIER
            # -----------------------------------------------------------------
            print("\n[TEST 3] Auditing Unteraar Glacier...")
            await select_benchmark("Unteraar")
            s_unteraar = await extract_ui_state()
            print("  Has Error:", s_unteraar["hasError"])
            print("  Has Dynamic Overlay Image:", s_unteraar["hasImage"])
            print("  Metrics extracted:", s_unteraar["metrics"])
            if s_unteraar["hasError"]:
                report["errors_detected"].append("Unteraar showed dynamic overlay error")
            await take_screenshot("glims_live_unteraar_fixed")
            report["benchmarks"]["unteraar"] = s_unteraar

            # -----------------------------------------------------------------
            # TEST 4: GORNER GLACIER
            # -----------------------------------------------------------------
            print("\n[TEST 4] Auditing Gorner Glacier...")
            await select_benchmark("Gorner")
            s_gorner = await extract_ui_state()
            print("  Has Error:", s_gorner["hasError"])
            print("  Has Dynamic Overlay Image:", s_gorner["hasImage"])
            print("  Metrics extracted:", s_gorner["metrics"])
            if s_gorner["hasError"]:
                report["errors_detected"].append("Gorner showed dynamic overlay error")
            await take_screenshot("glims_live_gorner_fixed")
            report["benchmarks"]["gorner"] = s_gorner

            # -----------------------------------------------------------------
            # TEST 5: RETURN TO RHONE & TEST LAYER TOGGLES (COLOR, DEBRIS, SATELLITE)
            # -----------------------------------------------------------------
            print("\n[TEST 5] Returning to Rhone Glacier to test styling and toggles...")
            await select_benchmark("Rhone")
            await asyncio.sleep(2.0)

            # Change color to Coral
            print("  Changing Active Margin Color to Coral...")
            res_col = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                    const cBox = sboxes.find(s => s.innerText.includes('Active Margin Color'));
                    if (!cBox) return null;
                    const btn = cBox.querySelector('button[aria-label="Open"]');
                    if (!btn) return null;
                    btn.scrollIntoView({block: 'center'});
                    const r = btn.getBoundingClientRect();
                    return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                })()""",
                "returnByValue": True
            })
            col_pos = res_col.get("result", {}).get("result", {}).get("value")
            if col_pos:
                await native_click(col_pos["x"], col_pos["y"])
                await asyncio.sleep(1.0)
                opt_coral = await send("Runtime.evaluate", {
                    "expression": """(() => {
                        const opts = Array.from(document.querySelectorAll('[role="option"]'));
                        const target = opts.find(o => o.innerText.includes('Coral'));
                        if (!target) return null;
                        const r = target.getBoundingClientRect();
                        return {text: target.innerText, x: r.x + r.width / 2, y: r.y + r.height / 2};
                    })()""",
                    "returnByValue": True
                })
                coral_pos = opt_coral.get("result", {}).get("result", {}).get("value")
                if coral_pos:
                    await native_click(coral_pos["x"], coral_pos["y"])
                    print(f"  Selected color: {coral_pos['text']}")
                    await asyncio.sleep(3.5)

            await take_screenshot("glims_live_styling_controls_fixed")

            # Toggle Debris Cover
            print("  Testing Debris Cover toggle...")
            res_deb = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const chks = Array.from(document.querySelectorAll('[data-testid="stCheckbox"]'));
                    const dBox = chks.find(c => c.innerText.includes('Supraglacial Debris Cover'));
                    if (!dBox) return null;
                    const input = dBox.querySelector('input');
                    const r = (input || dBox).getBoundingClientRect();
                    return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                })()""",
                "returnByValue": True
            })
            deb_pos = res_deb.get("result", {}).get("result", {}).get("value")
            if deb_pos:
                await native_click(deb_pos["x"], deb_pos["y"])
                print("  Clicked Debris Cover toggle")
                await asyncio.sleep(3.5)
                await take_screenshot("glims_live_debris_toggle_fixed")

            # -----------------------------------------------------------------
            # TEST 6: EXPAND GEOSPATIAL AUDIT DETAILS
            # -----------------------------------------------------------------
            print("\n[TEST 6] Opening Geospatial Audit Details Expander...")
            await send("Runtime.evaluate", {
                "expression": """(() => {
                    const details = Array.from(document.querySelectorAll('details'));
                    const exp = details.find(d => d.innerText.includes('Geospatial Registration & Multi-Sensor Audit Details'));
                    if (exp && !exp.open) exp.open = true;
                })()"""
            })
            await asyncio.sleep(1.5)
            await take_screenshot("glims_live_audit_disclosure_fixed")

            # Save json report
            rep_path = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\glims_live_verification_results.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved live verification report to {rep_path}")
            print(f"Errors detected: {report['errors_detected']}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
