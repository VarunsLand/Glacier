"""
verify_five_methods_cdp_live.py
Complete live browser verification of all five Landsat methods on sample 1_10_1.
"""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9565
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
AUDIT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits")
BRAIN_DIR = Path(r"C:\Users\Varun's LOQ\.gemini\antigravity\brain\4ab4cfe4-3217-44c3-a812-2c996eb7e1da\screenshots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

prof = OUTPUT_DIR / f"prof_methods_live_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--window-size=1440,2600",
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
        "methods": {},
        "round_trip_test": {},
        "panel_invariance": {},
    }

    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=3)
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
                "height": 2600,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(4.0)

            # 1. Click Landsat tab
            print("[1] Navigating to Landsat tab...")
            await send("Runtime.evaluate", {
                "expression": """(() => {
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const lTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                    if (lTab) lTab.click();
                })()"""
            })
            await asyncio.sleep(3.5)

            # Native click helper
            async def native_click(x, y):
                await send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": int(x), "y": int(y)})
                await send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": int(x), "y": int(y), "button": "left", "clickCount": 1})
                await send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": int(x), "y": int(y), "button": "left", "clickCount": 1})

            # Function to select method
            async def select_method(target_prefix):
                # 1. Scroll algorithm selectbox into view and get open button coords
                res = await send("Runtime.evaluate", {
                    "expression": """(() => {
                        const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                        const algoBox = sboxes.find(s => s.innerText.includes('Select Change Detection Algorithm'));
                        if (!algoBox) return null;
                        const btn = algoBox.querySelector('button[aria-label="Open"]');
                        if (!btn) return null;
                        btn.scrollIntoView({block: 'center'});
                        const r = btn.getBoundingClientRect();
                        return {x: r.x + r.width / 2, y: r.y + r.height / 2};
                    })()""",
                    "returnByValue": True
                })
                btn_pos = res.get("result", {}).get("result", {}).get("value")
                if not btn_pos:
                    raise RuntimeError("Could not find selectbox Open button")

                # Native click to open dropdown
                await native_click(btn_pos["x"], btn_pos["y"])
                await asyncio.sleep(1.0)

                # Find option position
                opt_res = await send("Runtime.evaluate", {
                    "expression": f"""(() => {{
                        const opts = Array.from(document.querySelectorAll('[role="option"]'));
                        const target = opts.find(o => o.innerText.includes('{target_prefix}'));
                        if (!target) return null;
                        const r = target.getBoundingClientRect();
                        return {{text: target.innerText, x: r.x + r.width / 2, y: r.y + r.height / 2}};
                    }})()""",
                    "returnByValue": True
                })
                opt_pos = opt_res.get("result", {}).get("result", {}).get("value")
                if not opt_pos:
                    raise RuntimeError(f"Could not find option with prefix {target_prefix}")

                # Click option
                await native_click(opt_pos["x"], opt_pos["y"])
                print(f"  Selected: {opt_pos['text']}")
                # Wait for Streamlit re-render
                await asyncio.sleep(3.5)

            # Function to extract current UI state
            async def extract_ui_state():
                extract_js = """(() => {
                    const text = document.body.innerText;
                    
                    // Metrics
                    const getMetric = (label) => {
                        const divs = Array.from(document.querySelectorAll('[data-testid="stMetric"]'));
                        const m = divs.find(d => d.innerText.includes(label));
                        return m ? m.innerText.split('\\n')[1] : null;
                    };
                    
                    // Slider
                    const sliders = Array.from(document.querySelectorAll('[data-testid="stSlider"]'));
                    const thSlider = sliders.find(s => s.innerText.includes('Deterministic Threshold'));
                    let sliderVal = null;
                    let sliderLabel = null;
                    if (thSlider) {
                        sliderLabel = thSlider.querySelector('label') ? thSlider.querySelector('label').innerText : null;
                        const lines = thSlider.innerText.split('\\n').filter(l => l.trim().length > 0);
                        sliderVal = lines.length > 1 ? lines[1] : null;
                    }

                    // Panels
                    const panels = [];
                    // Look for images in the 6-panel suite
                    const imgs = Array.from(document.querySelectorAll('[data-testid="stImage"] img'));
                    imgs.forEach((img, i) => {
                        const captionEl = img.parentElement ? img.parentElement.querySelector('figcaption, [data-testid="stCaptionContainer"]') : null;
                        panels.push({
                            index: i,
                            src_len: img.src ? img.src.length : 0,
                            src_prefix: img.src ? img.src.substring(0, 40) : '',
                            caption: captionEl ? captionEl.innerText : '',
                        });
                    });

                    // Scientific analysis block
                    const analysisCard = document.querySelector('.glass-card p');
                    const analysisText = analysisCard ? analysisCard.innerText : '';

                    return {
                        total_valid_pixels: getMetric('Total Valid Pixels'),
                        detected_changed_pixels: getMetric('Detected Changed Pixels'),
                        surface_change_ratio: getMetric('Surface Change Ratio'),
                        footprint: getMetric('Estimated Detected Pixel-Change Footprint'),
                        precision: getMetric('Precision (PPV)'),
                        recall: getMetric('Recall (Sensitivity)'),
                        specificity: getMetric('Specificity (TNR)'),
                        iou: getMetric('Intersection over Union (IoU)'),
                        dice: getMetric('Dice Score (F1)'),
                        slider_label: sliderLabel,
                        slider_val: sliderVal,
                        panels: panels,
                        analysis_snippet: analysisText.substring(0, 150),
                        raw_text_has_tp_2649: text.includes('2,649'),
                        raw_text_has_tp_2403: text.includes('2,403'),
                        raw_text_has_tp_775: text.includes('775'),
                        raw_text_has_tp_1149: text.includes('1,149'),
                    };
                })()"""
                r = await send("Runtime.evaluate", {"expression": extract_js, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value", {})

            # Method sequence
            methods = [
                ("Method A", "Method A: Baseline Mean RGB", "landsat_live_method_a.png"),
                ("Method B", "Method B: Grayscale Difference", "landsat_live_method_b.png"),
                ("Method C", "Method C: Calibrated Mean RGB", "landsat_live_method_c.png"),
                ("Method D", "Method D: Euclidean RGB", "landsat_live_method_d.png"),
                ("Method E", "Method E: Illumination Offset-Normalized", "landsat_live_method_e.png"),
            ]

            panel_history = {}

            for m_key, m_prefix, s_name in methods:
                print(f"\n[Test] Testing {m_key}...")
                await select_method(m_key)
                state = await extract_ui_state()

                # Take screenshot
                shot = await send("Page.captureScreenshot", {"format": "png"})
                s_bytes = base64.b64decode(shot.get("result", {}).get("data", ""))
                out_p = OUTPUT_DIR / s_name
                out_p.write_bytes(s_bytes)
                (BRAIN_DIR / s_name).write_bytes(s_bytes)

                report["methods"][m_key] = {
                    "prefix": m_prefix,
                    "screenshot": s_name,
                    "screenshot_bytes": len(s_bytes),
                    "state": state,
                }
                panel_history[m_key] = state.get("panels", [])

                print(f"  Threshold slider: {state.get('slider_label')} -> {state.get('slider_val')}")
                print(f"  Changed pixels: {state.get('detected_changed_pixels')}")
                print(f"  Precision: {state.get('precision')}, Recall: {state.get('recall')}, IoU: {state.get('iou')}, Dice: {state.get('dice')}")
                print(f"  Captured screenshot: {s_name} ({len(s_bytes):,} bytes)")

            # Round trip test: switch back to Method A
            print("\n[Round-Trip Test] Switching from Method E back to Method A...")
            await select_method("Method A")
            state_a_restored = await extract_ui_state()

            shot_r = await send("Page.captureScreenshot", {"format": "png"})
            s_bytes_r = base64.b64decode(shot_r.get("result", {}).get("data", ""))
            out_pr = OUTPUT_DIR / "landsat_live_method_a_restored.png"
            out_pr.write_bytes(s_bytes_r)
            (BRAIN_DIR / "landsat_live_method_a_restored.png").write_bytes(s_bytes_r)

            report["round_trip_test"] = {
                "restored_changed_pixels": state_a_restored.get("detected_changed_pixels"),
                "restored_precision": state_a_restored.get("precision"),
                "restored_recall": state_a_restored.get("recall"),
                "restored_iou": state_a_restored.get("iou"),
                "restored_dice": state_a_restored.get("dice"),
                "restored_screenshot": "landsat_live_method_a_restored.png",
                "matches_original_method_a": (
                    state_a_restored.get("detected_changed_pixels") == report["methods"]["Method A"]["state"].get("detected_changed_pixels")
                    and state_a_restored.get("precision") == report["methods"]["Method A"]["state"].get("precision")
                )
            }
            print(f"  Restored Method A changed pixels: {state_a_restored.get('detected_changed_pixels')}")
            print(f"  Restored Method A precision: {state_a_restored.get('precision')}")
            print(f"  Matches original Method A: {report['round_trip_test']['matches_original_method_a']}")

            # Save report
            out_json = AUDIT_DIR / "objective4_five_methods_live_audit.json"
            out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nAudit report successfully saved to: {out_json}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
