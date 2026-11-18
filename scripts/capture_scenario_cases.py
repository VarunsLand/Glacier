"""Capture clean scenario screenshots for Expansion and Retreat."""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")


async def capture_scenarios():
    port = 9492
    prof = OUTPUT_DIR / f"prof_scen_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        f"--user-data-dir={prof}",
        "http://localhost:8501/?page=area"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(2.0)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")
        targets = json.loads(resp.read().decode())
        page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")
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

            async def eval_js(expr):
                r = await send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value")

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 1300,
                "deviceScaleFactor": 1,
                "mobile": False,
            })

            for _ in range(40):
                ready = await eval_js("document.querySelectorAll('.kpi-card').length >= 5")
                if ready:
                    break
                await asyncio.sleep(0.3)
            await asyncio.sleep(1.0)

            # Test A: Expansion (A1=100, A2=110, dt=10)
            # Use Streamlit input update logic:
            set_script_exp = """
            (() => {
                const inputs = document.querySelectorAll('input[type="number"]');
                if (inputs.length < 3) return false;
                const setV = (input, val) => {
                    const desc = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                    desc.set.call(input, String(val));
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                };
                setV(inputs[0], 100);
                setV(inputs[1], 110);
                setV(inputs[2], 10);
                // Trigger form submission / blur
                inputs[2].focus();
                inputs[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
                inputs[2].blur();
                return true;
            })()
            """
            await eval_js(set_script_exp)
            await asyncio.sleep(3.0)

            cards_exp = await eval_js("""
                (() => {
                    const grids = document.querySelectorAll('.kpi-grid-4');
                    if (grids.length === 0) return [];
                    return Array.from(grids[0].querySelectorAll('.kpi-card')).map(c => ({
                        label: c.querySelector('.kpi-label')?.innerText?.trim(),
                        val: c.querySelector('.kpi-value')?.innerText?.trim(),
                        badge: c.querySelector('.kpi-badge')?.innerText?.trim()
                    }));
                })()
            """)
            print("Expansion Cards:", cards_exp)
            ss_exp = await send_cmd("Page.captureScreenshot", {"format": "png"})
            (OUTPUT_DIR / "verify_obj3_scenario_expansion.png").write_bytes(base64.b64decode(ss_exp["result"]["data"]))

            # Test B: Retreat (A1=100, A2=90, dt=10)
            set_script_ret = """
            (() => {
                const inputs = document.querySelectorAll('input[type="number"]');
                if (inputs.length < 3) return false;
                const setV = (input, val) => {
                    const desc = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                    desc.set.call(input, String(val));
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                };
                setV(inputs[0], 100);
                setV(inputs[1], 90);
                setV(inputs[2], 10);
                inputs[2].focus();
                inputs[2].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
                inputs[2].blur();
                return true;
            })()
            """
            await eval_js(set_script_ret)
            await asyncio.sleep(3.0)

            cards_ret = await eval_js("""
                (() => {
                    const grids = document.querySelectorAll('.kpi-grid-4');
                    if (grids.length === 0) return [];
                    return Array.from(grids[0].querySelectorAll('.kpi-card')).map(c => ({
                        label: c.querySelector('.kpi-label')?.innerText?.trim(),
                        val: c.querySelector('.kpi-value')?.innerText?.trim(),
                        badge: c.querySelector('.kpi-badge')?.innerText?.trim()
                    }));
                })()
            """)
            print("Retreat Cards:", cards_ret)
            ss_ret = await send_cmd("Page.captureScreenshot", {"format": "png"})
            (OUTPUT_DIR / "verify_obj3_scenario_retreat.png").write_bytes(base64.b64decode(ss_ret["result"]["data"]))

            print("Captured scenario screenshots successfully!")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(capture_scenarios())
