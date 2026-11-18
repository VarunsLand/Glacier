"""
test_change_algorithm_select.py
Test selecting each method in stSelectbox 6.
"""
import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9548
prof = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots") / f"prof_{PORT}"
prof.mkdir(parents=True, exist_ok=True)

cmd = [
    CHROME_PATH,
    "--headless=new",
    f"--remote-debugging-port={PORT}",
    "--disable-gpu",
    "--disable-extensions",
    f"--user-data-dir={prof}",
    "http://localhost:8501/?page=visual"
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3.0)

async def main():
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
            await asyncio.sleep(4.0)

            # Click Landsat tab
            await send("Runtime.evaluate", {
                "expression": """(() => {
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const lTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                    if (lTab) lTab.click();
                })()"""
            })
            await asyncio.sleep(3.0)

            # Function to select an option in Select Change Detection Algorithm
            test_select_fn = """(() => {
                const sboxes = Array.from(document.querySelectorAll('[data-testid="stSelectbox"]'));
                const algoBox = sboxes.find(s => s.innerText.includes('Select Change Detection Algorithm'));
                if (!algoBox) return {error: 'algoBox not found'};
                const input = algoBox.querySelector('input');
                if (!input) return {error: 'input not found in algoBox'};
                input.click();
                return {clicked: true};
            })()"""
            r1 = await send("Runtime.evaluate", {"expression": test_select_fn, "returnByValue": True})
            print("Click input:", r1.get("result", {}).get("result", {}).get("value", {}))
            await asyncio.sleep(1.0)

            # Find options
            check_opts = """(() => {
                const opts = Array.from(document.querySelectorAll('li[role="option"]'));
                return opts.map(o => o.innerText);
            })()"""
            r2 = await send("Runtime.evaluate", {"expression": check_opts, "returnByValue": True})
            print("Options found:", r2.get("result", {}).get("result", {}).get("value", []))

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
