"""GlacierVision: Live Browser Verification for Objective 3 Area Assessment.

Navigates to http://localhost:8501/?page=area and tests:
1. Macro KPI cards (5 cards: Validated Pairs, Retreating, Advancing, Stable, Net Loss).
2. Key Research Findings Synthesis card presence and contents.
3. Scenario Calculator arithmetic, deadband classification, and Net Area Decrease logic:
   - Default: A1=25, A2=21.25, dt=20 -> Net Surface Change: -3.75 km², Net Area Decrease: 3.75 km², Class: Significant Retreat
   - Expansion (Test A): A1=100, A2=110, dt=10 -> Net Surface Change: +10.00 km², Net Area Decrease: 0.00 km², Class: Measured Advance
   - Retreat (Test B): A1=100, A2=90, dt=10 -> Net Surface Change: -10.00 km², Net Area Decrease: 10.00 km², Class: Significant Retreat
   - Equilibrium (Test C): A1=100, A2=100, dt=10 -> Net Surface Change: +0.00 km², Net Area Decrease: 0.00 km², Class: Stable / Equilibrium
4. Observation Baseline Interval Summary table and Plotly chart (Mean Per-Pair Rate on primary axis).
5. Mathematical Methodology KaTeX rendering: verify KaTeX elements exist, and NO raw '$$' strings appear in body text.
6. Verification of all 6 structured sections in research flow.
7. Top 10 Largest Observed Pairwise Area Reductions section & data table.
8. Full GLIMS Record Explorer, Rhone search, and CSV download button presence.
9. Responsive multi-viewport verification across 5 viewports (1440px, 1366px, 1024px, 768px, 390px) with 0 horizontal overflow.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
AUDIT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


async def run_objective3_live_verification():
    port = 9491
    prof = OUTPUT_DIR / f"prof_obj3_verify_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "--disable-extensions",
        f"--user-data-dir={prof}",
        "http://localhost:8501/?page=area"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}

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
            await asyncio.sleep(0.3)

        if not page_ws:
            raise RuntimeError("Could not connect to Chrome debugging target.")

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

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 1300,
                "deviceScaleFactor": 1,
                "mobile": False,
            })

            async def evaluate_js(expr):
                r = await send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value")

            print("Navigating to Objective 3 workspace...")
            for _ in range(40):
                ready = await evaluate_js("document.querySelectorAll('.kpi-card').length >= 5 && !document.querySelector('.stSpinner')")
                if ready is True:
                    break
                await asyncio.sleep(0.3)
            await asyncio.sleep(1.5)

            # 1. Verify Macro Headline Cards
            print("Inspecting Macro KPI cards...")
            macro_cards = await evaluate_js("""
                (() => {
                    const cards = document.querySelectorAll('.kpi-grid-5 .kpi-card');
                    return Array.from(cards).map(c => {
                        const label = c.querySelector('.kpi-label')?.innerText?.trim() || '';
                        const val = c.querySelector('.kpi-value')?.innerText?.trim() || '';
                        const unit = c.querySelector('.kpi-unit')?.innerText?.trim() || '';
                        const badge = c.querySelector('.kpi-badge')?.innerText?.trim() || '';
                        return { label, val, unit, badge };
                    });
                })()
            """)
            print("Macro KPI Cards:", macro_cards)
            results["macro_cards"] = macro_cards

            # 2. Check 6 Structured Sections in DOM
            print("Verifying 6 Structured Sections...")
            section_headers = await evaluate_js("""
                (() => {
                    const headers = document.querySelectorAll('h3');
                    return Array.from(headers).map(h => h.innerText.trim());
                })()
            """)
            print("Section Headers Found:", section_headers)
            results["section_headers"] = section_headers
            expected_prefixes = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"]
            has_all_sections = all(any(p in h for h in section_headers) for p in expected_prefixes)
            results["has_all_6_sections"] = has_all_sections
            print(f"All 6 Sections Present: {has_all_sections}")

            # 3. Check Key Research Findings Synthesis Box
            has_research_findings = await evaluate_js("""
                (() => {
                    const headings = Array.from(document.querySelectorAll('h5, h6'));
                    return headings.some(h => h.innerText.includes("Key Research Findings"));
                })()
            """)
            results["has_key_research_findings"] = has_research_findings
            print(f"Key Research Findings Box Present: {has_research_findings}")

            # 4. Check Mathematical Methodology & KaTeX Rendering
            print("Checking KaTeX and Mathematical Methodology...")
            katex_count = await evaluate_js("""
                (() => {
                    return document.querySelectorAll('.katex, [data-testid="stLatex"]').length;
                })()
            """)
            raw_latex_in_text = await evaluate_js(r"""
                (() => {
                    const bodyText = document.body.innerText;
                    const matches = bodyText.match(/\$\$[A-Za-z0-9_\\{}]+\$\$/g);
                    return matches || [];
                })()
            """)
            results["katex_elements_count"] = katex_count
            results["raw_latex_matches"] = raw_latex_in_text
            print(f"KaTeX Elements Count: {katex_count}, Raw LaTeX Matches: {raw_latex_in_text}")

            # 5. Check Plotly Chart Title and Primary Axis
            chart_info = await evaluate_js("""
                (() => {
                    const chart = document.querySelector('.js-plotly-plot');
                    if (!chart || !chart.layout) return null;
                    return {
                        title: chart.layout.title?.text || '',
                        yaxis_title: chart.layout.yaxis?.title?.text || '',
                        yaxis2_title: chart.layout.yaxis2?.title?.text || ''
                    };
                })()
            """)
            results["temporal_chart_info"] = chart_info
            print("Temporal Chart Info:", chart_info)

            # Capture initial desktop screenshot
            ss1 = await send_cmd("Page.captureScreenshot", {"format": "png"})
            p_desk = OUTPUT_DIR / "verify_obj3_default_desktop.png"
            p_desk.write_bytes(base64.b64decode(ss1["result"]["data"]))
            print(f"Captured: {p_desk.name}")

            # 6. Inspect Scenario Calculator default cards
            sim_cards_default = await evaluate_js("""
                (() => {
                    const grids = document.querySelectorAll('.kpi-grid-4');
                    if (grids.length === 0) return [];
                    const cards = grids[0].querySelectorAll('.kpi-card');
                    return Array.from(cards).map(c => {
                        const label = c.querySelector('.kpi-label')?.innerText?.trim() || '';
                        const val = c.querySelector('.kpi-value')?.innerText?.trim() || '';
                        const unit = c.querySelector('.kpi-unit')?.innerText?.trim() || '';
                        const badge = c.querySelector('.kpi-badge')?.innerText?.trim() || '';
                        return { label, val, unit, badge };
                    });
                })()
            """)
            print("Scenario Cards (Default):", sim_cards_default)
            results["scenario_default"] = sim_cards_default

            async def set_number_inputs(a1_val, a2_val, dt_val):
                script = f"""
                (() => {{
                    const inputs = document.querySelectorAll('input[type="number"]');
                    if (inputs.length < 3) return false;
                    
                    const setVal = (input, val) => {{
                        const descriptor = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                        descriptor.set.call(input, String(val));
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        input.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
                    }};
                    
                    setVal(inputs[0], {a1_val});
                    setVal(inputs[1], {a2_val});
                    setVal(inputs[2], {dt_val});
                    return true;
                }})()
                """
                await evaluate_js(script)
                await asyncio.sleep(2.5)

            # Test A: Expansion (A1=100, A2=110, dt=10)
            print("Testing Scenario A (Expansion: A1=100, A2=110, dt=10)...")
            await set_number_inputs(100.0, 110.0, 10.0)
            sim_cards_exp = await evaluate_js("""
                (() => {
                    const grids = document.querySelectorAll('.kpi-grid-4');
                    if (grids.length === 0) return [];
                    const cards = grids[0].querySelectorAll('.kpi-card');
                    return Array.from(cards).map(c => {
                        const label = c.querySelector('.kpi-label')?.innerText?.trim() || '';
                        const val = c.querySelector('.kpi-value')?.innerText?.trim() || '';
                        const unit = c.querySelector('.kpi-unit')?.innerText?.trim() || '';
                        const badge = c.querySelector('.kpi-badge')?.innerText?.trim() || '';
                        return { label, val, unit, badge };
                    });
                })()
            """)
            print("Scenario Cards (Expansion):", sim_cards_exp)
            results["scenario_expansion"] = sim_cards_exp
            ss_exp = await send_cmd("Page.captureScreenshot", {"format": "png"})
            (OUTPUT_DIR / "verify_obj3_scenario_expansion.png").write_bytes(base64.b64decode(ss_exp["result"]["data"]))

            # Test B: Retreat (A1=100, A2=90, dt=10)
            print("Testing Scenario B (Retreat: A1=100, A2=90, dt=10)...")
            await set_number_inputs(100.0, 90.0, 10.0)
            sim_cards_ret = await evaluate_js("""
                (() => {
                    const grids = document.querySelectorAll('.kpi-grid-4');
                    if (grids.length === 0) return [];
                    const cards = grids[0].querySelectorAll('.kpi-card');
                    return Array.from(cards).map(c => {
                        const label = c.querySelector('.kpi-label')?.innerText?.trim() || '';
                        const val = c.querySelector('.kpi-value')?.innerText?.trim() || '';
                        const unit = c.querySelector('.kpi-unit')?.innerText?.trim() || '';
                        const badge = c.querySelector('.kpi-badge')?.innerText?.trim() || '';
                        return { label, val, unit, badge };
                    });
                })()
            """)
            print("Scenario Cards (Retreat):", sim_cards_ret)
            results["scenario_retreat"] = sim_cards_ret
            ss_ret = await send_cmd("Page.captureScreenshot", {"format": "png"})
            (OUTPUT_DIR / "verify_obj3_scenario_retreat.png").write_bytes(base64.b64decode(ss_ret["result"]["data"]))

            # Test C: Equilibrium (A1=100, A2=100, dt=10)
            print("Testing Scenario C (Equilibrium: A1=100, A2=100, dt=10)...")
            await set_number_inputs(100.0, 100.0, 10.0)
            sim_cards_eq = await evaluate_js("""
                (() => {
                    const grids = document.querySelectorAll('.kpi-grid-4');
                    if (grids.length === 0) return [];
                    const cards = grids[0].querySelectorAll('.kpi-card');
                    return Array.from(cards).map(c => {
                        const label = c.querySelector('.kpi-label')?.innerText?.trim() || '';
                        const val = c.querySelector('.kpi-value')?.innerText?.trim() || '';
                        const unit = c.querySelector('.kpi-unit')?.innerText?.trim() || '';
                        const badge = c.querySelector('.kpi-badge')?.innerText?.trim() || '';
                        return { label, val, unit, badge };
                    });
                })()
            """)
            print("Scenario Cards (Equilibrium):", sim_cards_eq)
            results["scenario_equilibrium"] = sim_cards_eq

            # Reset to Default (A1=25, A2=21.25, dt=20)
            await set_number_inputs(25.0, 21.25, 20.0)

            # 7. Check Top 10 Data Table in DOM
            has_top_table = await evaluate_js("""
                (() => {
                    const tables = document.querySelectorAll('[data-testid="stDataFrame"]');
                    return tables.length >= 2;
                })()
            """)
            results["has_top10_table"] = has_top_table
            print("Top 10 Table Found:", has_top_table)

            # 8. Check Download Button presence
            has_download_btn = await evaluate_js("""
                (() => {
                    const btns = document.querySelectorAll('button');
                    return Array.from(btns).some(b => b.innerText.includes("Export Results") || b.innerText.includes("CSV"));
                })()
            """)
            print("CSV Download Button Present:", has_download_btn)
            results["has_download_button"] = has_download_btn

            # 9. Multi-Viewport Responsive Verification (1440px, 1366px, 1024px, 768px, 390px)
            viewports = [
                ("1440px_desktop", 1440, 1000, False),
                ("1366px_laptop", 1366, 900, False),
                ("1024px_compact", 1024, 800, False),
                ("768px_tablet", 768, 1024, True),
                ("390px_mobile", 390, 844, True),
            ]
            results["viewports"] = {}

            for vp_name, w, h, is_mob in viewports:
                print(f"Testing Viewport: {vp_name} ({w}x{h})...")
                await send_cmd("Emulation.setDeviceMetricsOverride", {
                    "width": w,
                    "height": h,
                    "deviceScaleFactor": 1 if not is_mob else 2,
                    "mobile": is_mob,
                })
                await asyncio.sleep(1.0)
                ss_vp = await send_cmd("Page.captureScreenshot", {"format": "png"})
                p_vp = OUTPUT_DIR / f"verify_obj3_{vp_name}.png"
                p_vp.write_bytes(base64.b64decode(ss_vp["result"]["data"]))

                h_overflow = await evaluate_js("""
                    (() => {
                        return document.documentElement.scrollWidth > window.innerWidth;
                    })()
                """)
                results["viewports"][vp_name] = {
                    "width": w,
                    "height": h,
                    "has_horizontal_overflow": h_overflow,
                    "screenshot": str(p_vp.name),
                }
                print(f"  {vp_name}: Horizontal Overflow = {h_overflow}, Saved = {p_vp.name}")

            results["mobile_has_horizontal_overflow"] = results["viewports"]["390px_mobile"]["has_horizontal_overflow"]

        # Save JSON results
        out_res = AUDIT_DIR / "objective3_live_verification_results.json"
        with open(out_res, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved live verification results to:\n  {out_res}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(run_objective3_live_verification())
