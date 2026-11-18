"""
verify_five_methods_live.py
Automated live verification of all five Landsat methods on sample 1_10_1 via Chrome DevTools Protocol.
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
PORT = 9535
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
AUDIT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits")
BRAIN_DIR = Path(r"C:\Users\Varun's LOQ\.gemini\antigravity\brain\4ab4cfe4-3217-44c3-a812-2c996eb7e1da\screenshots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

prof = OUTPUT_DIR / f"prof_landsat_methods_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--disable-gpu",
    "--disable-extensions",
    f"--user-data-dir={prof}",
    "http://localhost:8501/?page=monitoring"
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(2.5)

async def main():
    results = {}
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
                "height": 2200,
                "deviceScaleFactor": 1,
                "mobile": False
            })
            await asyncio.sleep(2.0)

            print("[1] Navigating to Landsat Tab...")
            click_tab = """(() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const landsatTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                if (landsatTab) { landsatTab.click(); return true; }
                return false;
            })()"""
            await send("Runtime.evaluate", {"expression": click_tab})
            await asyncio.sleep(3.5)

            # Helper to select a method from the algorithm method selectbox
            select_method_js = """((targetText) => {
                const labels = Array.from(document.querySelectorAll('label'));
                const methodLabel = labels.find(l => l.innerText && l.innerText.includes('Detection Algorithm'));
                if (!methodLabel) return {success: false, error: 'Method label not found'};
                const container = methodLabel.parentElement;
                const select = container.querySelector('div[data-baseweb="select"]');
                if (!select) return {success: false, error: 'Select box not found'};
                select.click();
                return {success: true};
            })"""

            pick_option_js = """((targetPrefix) => {
                const options = Array.from(document.querySelectorAll('li[role="option"]'));
                const opt = options.find(o => o.innerText && o.innerText.includes(targetPrefix));
                if (opt) {
                    opt.click();
                    return {success: true, selected: opt.innerText};
                }
                return {success: false, available: options.map(o => o.innerText)};
            })"""

            methods_to_test = [
                ("Method A", "Method A: Baseline Mean RGB", "landsat_live_method_a.png"),
                ("Method B", "Method B: Grayscale Difference", "landsat_live_method_b.png"),
                ("Method C", "Method C: Calibrated Mean RGB", "landsat_live_method_c.png"),
                ("Method D", "Method D: Euclidean RGB", "landsat_live_method_d.png"),
                ("Method E", "Method E: Illumination Offset-Normalized", "landsat_live_method_e.png"),
            ]

            panel_img_hashes = {}

            for m_key, m_prefix, screenshot_name in methods_to_test:
                print(f"\n[Test] Selecting {m_key} ({m_prefix})...")
                # Open dropdown
                r1 = await send("Runtime.evaluate", {"expression": f"({select_method_js})()", "returnByValue": True})
                await asyncio.sleep(0.8)
                # Pick option
                r2 = await send("Runtime.evaluate", {"expression": f"({pick_option_js})('{m_key}')", "returnByValue": True})
                # Wait for Streamlit re-render
                await asyncio.sleep(3.0)

                # Capture full text
                body = (await send("Runtime.evaluate", {"expression": "document.body.innerText"})).get("result", {}).get("result", {}).get("value", "")

                # Extract panel images src or lengths
                extract_panels = """(() => {
                    const imgs = Array.from(document.querySelectorAll('img'));
                    // Find images with captions in the 6-panel suite
                    const panels = [];
                    imgs.forEach((img, idx) => {
                        const caption = img.parentElement ? img.parentElement.innerText : '';
                        panels.push({
                            idx: idx,
                            src_len: img.src ? img.src.length : 0,
                            src_prefix: img.src ? img.src.substring(0, 50) : '',
                            alt: img.alt || '',
                        });
                    });
                    return panels;
                })()"""
                panels_data = (await send("Runtime.evaluate", {"expression": extract_panels, "returnByValue": True})).get("result", {}).get("result", {}).get("value", [])

                # Take screenshot
                shot = await send("Page.captureScreenshot", {"format": "png"})
                s_data = shot.get("result", {}).get("data", "")
                if s_data:
                    p = OUTPUT_DIR / screenshot_name
                    p.write_bytes(base64.b64decode(s_data))
                    (BRAIN_DIR / screenshot_name).write_bytes(base64.b64decode(s_data))
                    print(f"  Saved screenshot: {p.name}")

                # Check expected strings in text
                m_info = {
                    "method": m_key,
                    "body_snippet_sample": "1_10_1" in body,
                    "method_in_body": m_key in body,
                    "panels_count": len(panels_data),
                }

                # Extract specific numbers if present
                for val in ["2,649", "52,140", "4,145", "6,602", "2,403", "148", "4,391", "58,594", "775", "276", "1,149", "859", "2,641", "52,652"]:
                    if val in body:
                        m_info[f"has_{val.replace(',', '_')}"] = True

                results[m_key] = m_info
                panel_img_hashes[m_key] = panels_data
                print(f"  {m_key} verification info: {m_info}")

            # Test Round Trip: Switch A -> E -> A
            print("\n[Round-Trip Test] Switching back to Method A...")
            await send("Runtime.evaluate", {"expression": f"({select_method_js})()", "returnByValue": True})
            await asyncio.sleep(0.8)
            await send("Runtime.evaluate", {"expression": f"({pick_option_js})('Method A')", "returnByValue": True})
            await asyncio.sleep(3.0)

            body_restored = (await send("Runtime.evaluate", {"expression": "document.body.innerText"})).get("result", {}).get("result", {}).get("value", "")
            shot_restored = await send("Page.captureScreenshot", {"format": "png"})
            s_data_r = shot_restored.get("result", {}).get("data", "")
            if s_data_r:
                p_r = OUTPUT_DIR / "landsat_live_method_a_restored.png"
                p_r.write_bytes(base64.b64decode(s_data_r))
                (BRAIN_DIR / "landsat_live_method_a_restored.png").write_bytes(base64.b64decode(s_data_r))
                print(f"  Saved restored screenshot: {p_r.name}")

            restored_checks = {
                "tp_2649": "2,649" in body_restored,
                "fp_52140": "52,140" in body_restored,
                "fn_4145": "4,145" in body_restored,
                "tn_6602": "6,602" in body_restored,
                "precision_0_0483": "0.0483" in body_restored,
                "changed_54789": "54,789" in body_restored,
            }
            results["RoundTrip_Restored_Method_A"] = restored_checks
            print("  Restored Method A checks:", restored_checks)

            # Save full audit results
            res_path = AUDIT_DIR / "objective4_five_methods_live_audit.json"
            res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nAudit results written to: {res_path}")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
