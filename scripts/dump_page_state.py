"""
dump_page_state.py
Check title, radio buttons, tabs, and page text.
"""
import asyncio
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9542
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

            # Check document title and current page text
            info = await send("Runtime.evaluate", {
                "expression": "({title: document.title, url: window.location.href, textSnippet: document.body.innerText.substring(0, 500)})",
                "returnByValue": True
            })
            print("Page info:", json.dumps(info.get("result", {}).get("result", {}).get("value", {}), indent=2))

            # Check sidebar radio buttons
            radios = await send("Runtime.evaluate", {
                "expression": "Array.from(document.querySelectorAll('[role=\"radiogroup\"] label')).map(l => l.innerText)",
                "returnByValue": True
            })
            print("Radios:", json.dumps(radios.get("result", {}).get("result", {}).get("value", []), indent=2))

            # Check tabs
            tabs = await send("Runtime.evaluate", {
                "expression": "Array.from(document.querySelectorAll('[role=\"tab\"]')).map(t => t.innerText)",
                "returnByValue": True
            })
            print("Tabs:", json.dumps(tabs.get("result", {}).get("result", {}).get("value", []), indent=2))

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(main())
