"""
test_click_landsat_tab.py
Click Landsat tab and inspect the selectboxes and controls inside.
"""
import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9544
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
            click_code = """(() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const lTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                if (lTab) {
                    lTab.click();
                    return {clicked: true, text: lTab.innerText};
                }
                return {clicked: false};
            })()"""
            c_res = await send("Runtime.evaluate", {"expression": click_code, "returnByValue": True})
            print("Click tab result:", c_res.get("result", {}).get("result", {}).get("value", {}))
            await asyncio.sleep(4.0)

            # Inspect all selectboxes on page now
            dump_selects = """(() => {
                const results = [];
                const selects = document.querySelectorAll('div[data-baseweb="select"]');
                selects.forEach((s, idx) => {
                    let label = '';
                    let parent = s.parentElement;
                    for (let k = 0; k < 5; k++) {
                        if (!parent) break;
                        const l = parent.querySelector('label');
                        if (l) { label = l.innerText; break; }
                        parent = parent.parentElement;
                    }
                    results.push({
                        idx: idx,
                        label: label,
                        text: s.innerText,
                        id: s.id,
                    });
                });
                return results;
            })()"""
            s_res = await send("Runtime.evaluate", {"expression": dump_selects, "returnByValue": True})
            print("Selectboxes after tab click:", json.dumps(s_res.get("result", {}).get("result", {}).get("value", []), indent=2))

            # Also check sliders
            dump_sliders = """(() => {
                const results = [];
                const sliders = document.querySelectorAll('div[data-testid="stSlider"]');
                sliders.forEach((s, idx) => {
                    results.push({
                        idx: idx,
                        label: s.querySelector('label') ? s.querySelector('label').innerText : '',
                        value: s.innerText,
                    });
                });
                return results;
            })()"""
            sl_res = await send("Runtime.evaluate", {"expression": dump_sliders, "returnByValue": True})
            print("Sliders after tab click:", json.dumps(sl_res.get("result", {}).get("result", {}).get("value", []), indent=2))

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
