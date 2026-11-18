"""Live browser verification script for custom GLIMS ID resolution and visualization."""
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
AUDIT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

async def run_live_custom_glims_test():
    port = 9486
    prof = OUTPUT_DIR / f"prof_custom_glims_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "--disable-extensions",
        f"--user-data-dir={prof}",
        "http://localhost:8501/?page=retreat"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
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
                nonlocal mid; mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid: return d

            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 1300,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            async def wait_for_streamlit_ready():
                for _ in range(40):
                    ev = await send_cmd("Runtime.evaluate", {
                        "expression": "document.querySelectorAll('[data-testid=\"stMarkdown\"]').length > 3 && !document.querySelector('.stSpinner')",
                        "returnByValue": True
                    })
                    val = ev.get("result", {}).get("result", {}).get("value")
                    if val is True:
                        await asyncio.sleep(1.0)
                        return True
                    await asyncio.sleep(0.3)
                return False

            async def get_page_state():
                script = """
                (() => {
                    const inputEl = document.querySelector('input[data-testid="stTextInputRootElement"]') || 
                                    document.querySelector('input[aria-label="Search by Glacier ID or Benchmark Name:"]');
                    const searchVal = inputEl ? inputEl.value : "";
                    
                    const h4 = Array.from(document.querySelectorAll('h4')).find(el => el.textContent.includes('Observation Epochs'));
                    const headingText = h4 ? h4.textContent : "";
                    
                    let warningText = "";
                    document.querySelectorAll('[data-testid="stAlert"]').forEach(el => {
                        if (el.innerText.includes("not found")) {
                            warningText = el.innerText;
                        }
                    });
                    
                    let infoText = "";
                    document.querySelectorAll('[data-testid="stAlert"]').forEach(el => {
                        if (el.innerText.includes("Individual Geometry Visualization") || el.innerText.includes("Individual Boundary Map Status")) {
                            infoText = el.innerText;
                        }
                    });

                    let netChange = "";
                    let earlierArea = "";
                    let modernArea = "";
                    document.querySelectorAll('.kpi-card').forEach(c => {
                        const txt = c.textContent.toLowerCase();
                        if (txt.includes("net surface change")) {
                            const valEl = c.querySelector('.kpi-value');
                            if (valEl) netChange = valEl.textContent.trim();
                        }
                        if (txt.includes("earlier area")) {
                            const valEl = c.querySelector('.kpi-value');
                            if (valEl) earlierArea = valEl.textContent.trim();
                        }
                        if (txt.includes("modern area")) {
                            const valEl = c.querySelector('.kpi-value');
                            if (valEl) modernArea = valEl.textContent.trim();
                        }
                    });

                    let imgCaptions = [];
                    document.querySelectorAll('img').forEach(img => {
                        const cap = img.getAttribute('alt') || (img.parentElement ? img.parentElement.innerText : "");
                        const src = img.getAttribute('src') || "";
                        imgCaptions.push({src: src.substring(0, 80), cap: cap.substring(0, 100)});
                    });

                    return JSON.stringify({
                        searchVal,
                        headingText,
                        warningText,
                        infoText,
                        netChange,
                        earlierArea,
                        modernArea,
                        imgCaptions
                    });
                })()
                """
                res = await send_cmd("Runtime.evaluate", {"expression": script, "returnByValue": True})
                raw = res.get("result", {}).get("result", {}).get("value", "{}")
                return json.loads(raw)

            async def capture_screenshot(filename: str):
                ss = await send_cmd("Page.captureScreenshot", {"format": "png"})
                if "data" in ss.get("result", {}):
                    p = OUTPUT_DIR / filename
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(ss["result"]["data"]))
                    return str(p)
                return ""

            async def click_button_by_text(btn_text: str):
                click_js = f"""
                (() => {{
                    const buttons = Array.from(document.querySelectorAll('div.stButton > button'));
                    const target = buttons.find(b => {{
                        const p = b.querySelector('p');
                        return p && p.textContent.trim() === '{btn_text}';
                    }});
                    if (target) {{
                        target.click();
                        return true;
                    }}
                    return false;
                }})()
                """
                res = await send_cmd("Runtime.evaluate", {"expression": click_js, "returnByValue": True})
                return res.get("result", {}).get("result", {}).get("value", False)

            async def enter_search_text(text: str):
                input_js = f"""
                (() => {{
                    const inputEl = document.querySelector('input[data-testid="stTextInputRootElement"]') || 
                                    document.querySelector('input[aria-label="Search by Glacier ID or Benchmark Name:"]');
                    if (!inputEl) return false;
                    
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    nativeInputValueSetter.call(inputEl, '{text}');
                    
                    inputEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    inputEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    
                    inputEl.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
                    inputEl.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
                    return true;
                }})()
                """
                res = await send_cmd("Runtime.evaluate", {"expression": input_js, "returnByValue": True})
                return res.get("result", {}).get("result", {}).get("value", False)

            print("Waiting for Streamlit initial load...")
            await wait_for_streamlit_ready()

            # Step 1: Initial Benchmark (Rhone)
            state_1 = await get_page_state()
            ss_1 = await capture_screenshot("custom_test_1_rhone.png")
            results.append({"step": "1_Rhone", "state": state_1, "screenshot": ss_1})
            print("Step 1 (Rhone):", state_1.get("searchVal"), state_1.get("netChange"))

            # Step 2: Great Aletsch
            await click_button_by_text("Great Aletsch")
            await asyncio.sleep(2.0)
            await wait_for_streamlit_ready()
            state_2 = await get_page_state()
            ss_2 = await capture_screenshot("custom_test_2_great_aletsch.png")
            results.append({"step": "2_Great_Aletsch", "state": state_2, "screenshot": ss_2})
            print("Step 2 (Great Aletsch):", state_2.get("searchVal"), state_2.get("netChange"))

            # Step 3: Unteraar
            await click_button_by_text("Unteraar")
            await asyncio.sleep(2.0)
            await wait_for_streamlit_ready()
            state_3 = await get_page_state()
            ss_3 = await capture_screenshot("custom_test_3_unteraar.png")
            results.append({"step": "3_Unteraar", "state": state_3, "screenshot": ss_3})
            print("Step 3 (Unteraar):", state_3.get("searchVal"), state_3.get("netChange"))

            # Step 4: Gorner
            await click_button_by_text("Gorner")
            await asyncio.sleep(2.0)
            await wait_for_streamlit_ready()
            state_4 = await get_page_state()
            ss_4 = await capture_screenshot("custom_test_4_gorner.png")
            results.append({"step": "4_Gorner", "state": state_4, "screenshot": ss_4})
            print("Step 4 (Gorner):", state_4.get("searchVal"), state_4.get("netChange"))

            # Step 5: Valid Custom GLIMS ID with pre-rendered geometry (G010725E46789N - Mazia / Matscher)
            await enter_search_text("G010725E46789N")
            await asyncio.sleep(2.5)
            await wait_for_streamlit_ready()
            state_5 = await get_page_state()
            ss_5 = await capture_screenshot("custom_test_5_mazia_G010725E46789N.png")
            results.append({"step": "5_Mazia_G010725E46789N", "state": state_5, "screenshot": ss_5})
            print("Step 5 (Mazia G010725E46789N):", state_5.get("searchVal"), state_5.get("netChange"), state_5.get("earlierArea"), state_5.get("modernArea"))

            # Step 6: Valid Custom GLIMS ID without pre-rendered geometry (G057419E80930N)
            await enter_search_text("G057419E80930N")
            await asyncio.sleep(2.5)
            await wait_for_streamlit_ready()
            state_6 = await get_page_state()
            ss_6 = await capture_screenshot("custom_test_6_custom_no_map_G057419E80930N.png")
            results.append({"step": "6_Custom_No_Map_G057419E80930N", "state": state_6, "screenshot": ss_6})
            print("Step 6 (Custom No Map G057419E80930N):", state_6.get("searchVal"), state_6.get("netChange"), state_6.get("infoText")[:60] if state_6.get("infoText") else "None")

            # Step 7: Invalid GLIMS ID (INVALID_GLACIER_XYZ)
            await enter_search_text("INVALID_GLACIER_XYZ")
            await asyncio.sleep(2.5)
            await wait_for_streamlit_ready()
            state_7 = await get_page_state()
            ss_7 = await capture_screenshot("custom_test_7_invalid_query.png")
            results.append({"step": "7_Invalid_Query", "state": state_7, "screenshot": ss_7})
            print("Step 7 (Invalid Query):", state_7.get("searchVal"), state_7.get("warningText")[:60] if state_7.get("warningText") else "None")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    out_file = AUDIT_DIR / "custom_glims_live_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit results written to {out_file}")

if __name__ == "__main__":
    asyncio.run(run_live_custom_glims_test())
