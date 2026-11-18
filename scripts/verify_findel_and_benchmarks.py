"""Live browser verification for Findel and benchmarks post-unification."""
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

async def run_live_verification():
    port = 9488
    prof = OUTPUT_DIR / f"prof_findel_verify_{port}"
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

                    let netChange = "";
                    let earlierArea = "";
                    let modernArea = "";
                    let pctBadge = "";
                    document.querySelectorAll('.kpi-card').forEach(c => {
                        const txt = c.textContent.toLowerCase();
                        if (txt.includes("net surface change")) {
                            const valEl = c.querySelector('.kpi-value');
                            if (valEl) netChange = valEl.textContent.trim();
                            const bEl = c.querySelector('.kpi-badge');
                            if (bEl) pctBadge = bEl.textContent.trim();
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

                    let decompList = [];
                    document.querySelectorAll('ul li, ol li').forEach(li => {
                        const txt = li.textContent.trim();
                        if (txt.includes("Outline") || txt.includes("Core Ice") || txt.includes("Shrinkage")) {
                            decompList.push(txt);
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
                        netChange,
                        earlierArea,
                        modernArea,
                        pctBadge,
                        decompList,
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

            # Test 1: Rhone
            state_rhone = await get_page_state()
            ss_rhone = await capture_screenshot("verify_1_rhone.png")
            results.append({"glacier": "Rhone", "state": state_rhone, "screenshot": ss_rhone})
            print("Rhone:", state_rhone.get("earlierArea"), state_rhone.get("modernArea"), state_rhone.get("netChange"))

            # Test 2: Great Aletsch
            await click_button_by_text("Great Aletsch")
            await asyncio.sleep(2.0)
            await wait_for_streamlit_ready()
            state_aletsch = await get_page_state()
            ss_aletsch = await capture_screenshot("verify_2_great_aletsch.png")
            results.append({"glacier": "Great Aletsch", "state": state_aletsch, "screenshot": ss_aletsch})
            print("Great Aletsch:", state_aletsch.get("earlierArea"), state_aletsch.get("modernArea"), state_aletsch.get("netChange"))

            # Test 3: Unteraar
            await click_button_by_text("Unteraar")
            await asyncio.sleep(2.0)
            await wait_for_streamlit_ready()
            state_unteraar = await get_page_state()
            ss_unteraar = await capture_screenshot("verify_3_unteraar.png")
            results.append({"glacier": "Unteraar", "state": state_unteraar, "screenshot": ss_unteraar})
            print("Unteraar:", state_unteraar.get("earlierArea"), state_unteraar.get("modernArea"), state_unteraar.get("netChange"))

            # Test 4: Gorner
            await click_button_by_text("Gorner")
            await asyncio.sleep(2.0)
            await wait_for_streamlit_ready()
            state_gorner = await get_page_state()
            ss_gorner = await capture_screenshot("verify_4_gorner.png")
            results.append({"glacier": "Gorner", "state": state_gorner, "screenshot": ss_gorner})
            print("Gorner:", state_gorner.get("earlierArea"), state_gorner.get("modernArea"), state_gorner.get("netChange"))

            # Test 5: Findel (G007880E45990N)
            await enter_search_text("G007880E45990N")
            await asyncio.sleep(2.5)
            await wait_for_streamlit_ready()
            state_findel = await get_page_state()
            ss_findel = await capture_screenshot("verify_5_findel.png")
            results.append({"glacier": "Findel", "state": state_findel, "screenshot": ss_findel})
            print("Findel:", state_findel.get("earlierArea"), state_findel.get("modernArea"), state_findel.get("netChange"), state_findel.get("pctBadge"))

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    out_file = AUDIT_DIR / "findel_and_benchmarks_verification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    asyncio.run(run_live_verification())
