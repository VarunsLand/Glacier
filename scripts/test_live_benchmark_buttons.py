"""Live browser interaction test verifying benchmark buttons and manual search."""
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

async def run_live_test():
    port = 9480
    prof = OUTPUT_DIR / f"prof_button_test_{port}"
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
            await asyncio.sleep(0.2)

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
                "mobile": False
            })

            # Wait for Streamlit page to hydrate
            for _ in range(30):
                await asyncio.sleep(0.5)
                res = await send_cmd("Runtime.evaluate", {
                    "expression": "document.querySelectorAll('.kpi-card').length"
                })
                val = res.get("result", {}).get("result", {}).get("value", 0)
                if val >= 4:
                    break

            await asyncio.sleep(2.0)

            async def get_state():
                eval_res = await send_cmd("Runtime.evaluate", {
                    "expression": """
                    (() => {
                        const inputEl = document.querySelector('input[data-testid="stTextInputRootElement"]') || document.querySelector('input[aria-label="Search by Glacier ID or Benchmark Name:"]');
                        const searchVal = inputEl ? inputEl.value : '';
                        
                        // Extract GLIMS ID from header or parameters card
                        const h4 = Array.from(document.querySelectorAll('h4')).find(el => el.textContent.includes('Observation Epochs'));
                        const headingText = h4 ? h4.textContent : '';
                        
                        // Extract net area change from KPI cards
                        const cards = Array.from(document.querySelectorAll('.kpi-card'));
                        let netChange = '';
                        cards.forEach(c => {
                            if (c.textContent.toLowerCase().includes('net surface change')) {
                                const valEl = c.querySelector('.kpi-value');
                                if (valEl) netChange = valEl.textContent.trim();
                            }
                        });

                        // Extract which benchmark button has kind="primary"
                        const buttons = Array.from(document.querySelectorAll('div.stButton > button'));
                        const btnStates = {};
                        buttons.forEach(b => {
                            const p = b.querySelector('p');
                            if (p && ['Rhone', 'Great Aletsch', 'Unteraar', 'Gorner'].includes(p.textContent.trim())) {
                                btnStates[p.textContent.trim()] = b.getAttribute('kind') || 'none';
                            }
                        });

                        return JSON.stringify({
                            searchVal,
                            headingText,
                            netChange,
                            btnStates
                        });
                    })()
                    """
                })
                raw = eval_res.get("result", {}).get("result", {}).get("value", "{}")
                return json.loads(raw)

            async def click_button_by_text(btn_text):
                res = await send_cmd("Runtime.evaluate", {
                    "expression": f"""
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
                })
                return res.get("result", {}).get("result", {}).get("value", False)

            async def enter_search_text(text):
                res = await send_cmd("Runtime.evaluate", {
                    "expression": f"""
                    (() => {{
                        const inputEl = document.querySelector('input[data-testid="stTextInputRootElement"]') || document.querySelector('input[aria-label="Search by Glacier ID or Benchmark Name:"]');
                        if (!inputEl) return false;
                        
                        // Native value setter to simulate actual typing
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        nativeInputValueSetter.call(inputEl, '{text}');
                        
                        inputEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        inputEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        
                        // Dispatch Enter key
                        inputEl.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
                        inputEl.dispatchEvent(new KeyboardEvent('keyup', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
                        return true;
                    }})()
                    """
                })
                return res.get("result", {}).get("result", {}).get("value", False)

            async def capture_screen(filename):
                shot = await send_cmd("Page.captureScreenshot", {"format": "png"})
                out_p = OUTPUT_DIR / filename
                out_p.write_bytes(base64.b64decode(shot["result"]["data"]))
                print(f"Captured: {filename}")

            # Step 1: Initial Rhone State
            print("\n--- STEP 1: Initial Rhone State ---")
            st1 = await get_state()
            print("State:", st1)
            await capture_screen("btn_test_1_rhone_initial.png")
            results.append({"step": "1_Rhone_Initial", "state": st1})

            # Step 2: Click 'Great Aletsch'
            print("\n--- STEP 2: Click 'Great Aletsch' ---")
            clicked = await click_button_by_text("Great Aletsch")
            print("Button clicked:", clicked)
            # Wait for rerun
            await asyncio.sleep(2.5)
            st2 = await get_state()
            print("State:", st2)
            await capture_screen("btn_test_2_great_aletsch.png")
            results.append({"step": "2_Great_Aletsch", "state": st2})

            # Step 3: Click 'Unteraar'
            print("\n--- STEP 3: Click 'Unteraar' ---")
            clicked = await click_button_by_text("Unteraar")
            print("Button clicked:", clicked)
            await asyncio.sleep(2.5)
            st3 = await get_state()
            print("State:", st3)
            await capture_screen("btn_test_3_unteraar.png")
            results.append({"step": "3_Unteraar", "state": st3})

            # Step 4: Click 'Gorner'
            print("\n--- STEP 4: Click 'Gorner' ---")
            clicked = await click_button_by_text("Gorner")
            print("Button clicked:", clicked)
            await asyncio.sleep(2.5)
            st4 = await get_state()
            print("State:", st4)
            await capture_screen("btn_test_4_gorner.png")
            results.append({"step": "4_Gorner", "state": st4})

            # Step 5: Click 'Rhone'
            print("\n--- STEP 5: Click 'Rhone' (Return) ---")
            clicked = await click_button_by_text("Rhone")
            print("Button clicked:", clicked)
            await asyncio.sleep(2.5)
            st5 = await get_state()
            print("State:", st5)
            await capture_screen("btn_test_5_rhone_return.png")
            results.append({"step": "5_Rhone_Return", "state": st5})

            # Step 6: Custom Search Entry 'G007880E45990N' (Findel)
            print("\n--- STEP 6: Manual Search Entry 'G007880E45990N' ---")
            entered = await enter_search_text("G007880E45990N")
            print("Text entered:", entered)
            await asyncio.sleep(2.5)
            st6 = await get_state()
            print("State:", st6)
            await capture_screen("btn_test_6_manual_findel.png")
            results.append({"step": "6_Manual_Findel", "state": st6})

            # Save summary JSON
            res_file = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\benchmark_button_live_test_results.json")
            res_file.write_text(json.dumps(results, indent=2))
            print(f"\nSaved test results to {res_file}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(run_live_test())
