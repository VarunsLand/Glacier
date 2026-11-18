"""
inspect_streamlit_select.py
Inspect the exact structure of the algorithm selectbox in Streamlit.
"""
import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9540
prof = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots") / f"prof_{PORT}"
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
            await asyncio.sleep(2.0)

            # Click tab
            click_tab = """(() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const landsatTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat Bitemporal Change Explorer'));
                if (landsatTab) landsatTab.click();
            })()"""
            await send("Runtime.evaluate", {"expression": click_tab})
            await asyncio.sleep(3.5)

            # Dump all selectboxes and their labels
            dump_selects = """(() => {
                const results = [];
                const selects = document.querySelectorAll('div[data-baseweb="select"]');
                selects.forEach((s, idx) => {
                    let label = '';
                    let parent = s.parentElement;
                    while (parent && !label) {
                        const l = parent.querySelector('label');
                        if (l) label = l.innerText;
                        parent = parent.parentElement;
                    }
                    results.push({
                        idx: idx,
                        label: label,
                        current_text: s.innerText,
                        aria_expanded: s.querySelector('div[aria-expanded]') ? s.querySelector('div[aria-expanded]').getAttribute('aria-expanded') : null
                    });
                });
                return results;
            })()"""
            r = await send("Runtime.evaluate", {"expression": dump_selects, "returnByValue": True})
            print("Selectboxes found:", json.dumps(r.get("result", {}).get("result", {}).get("value", []), indent=2))

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
