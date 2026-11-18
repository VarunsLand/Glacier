import asyncio
import json
import subprocess
import urllib.request
import websockets

async def test_tab_click():
    port = 9496
    cmd = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "http://localhost:8501/?page=monitoring"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(2)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")
        targets = json.loads(resp.read().decode())
        ws_url = [t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page"][0]
        async with websockets.connect(ws_url) as ws:
            mid = 0
            async def call(m, p=None):
                nonlocal mid
                mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid:
                        return d
            await call("Runtime.enable")
            await asyncio.sleep(4)

            # Click Landsat tab
            click_res = await call("Runtime.evaluate", {
                "expression": """
                (() => {
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const lTab = tabs.find(t => t.innerText && t.innerText.includes('Landsat'));
                    if (lTab) {
                        lTab.click();
                        return true;
                    }
                    return false;
                })()
                """
            })
            print("Landsat tab clicked:", click_res.get("result", {}).get("result", {}).get("value"))
            await asyncio.sleep(3)

            # Get body text
            body_res = await call("Runtime.evaluate", {"expression": "document.body.innerText"})
            text = body_res.get("result", {}).get("result", {}).get("value", "")
            print("Contains Earlier Observation (A):", "Earlier Observation (A)" in text or "Earlier Observation" in text)
            print("Contains Later Observation (B):", "Later Observation (B)" in text or "Later Observation" in text)
            print("Contains Absolute Pixel Deviation:", "Absolute Pixel Deviation" in text or "Difference Magnitude" in text)
            print("Contains Binary Detected-Change Mask:", "Binary Detected-Change Mask" in text or "Change Mask" in text)
            print("Contains Coral Red Overlay:", "Coral Red Overlay" in text or "Coral" in text)
            print("Contains Scientific Methodology & Disclaimer:", "Scientific Methodology & Disclaimer" in text or "undirected optical difference" in text)
            print("Contains Ground-Truth Reference Evaluation:", "Ground-Truth Reference Evaluation" in text or "Intersection over Union" in text)
    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(test_tab_click())
