import asyncio
import json
import subprocess
import urllib.request
import websockets

async def check():
    port = 9501
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
                nonlocal mid; mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid: return d
            await call("Runtime.enable")
            await call("DOM.enable")
            await asyncio.sleep(3)

            # Ensure on Objective 4
            await call("Runtime.evaluate", {"expression": """
                (() => {
                    const radios = Array.from(document.querySelectorAll('input[name="Navigation Menu"]'));
                    // or sidebar labels
                    const labels = Array.from(document.querySelectorAll('label'));
                    const visLabel = labels.find(l => l.innerText && l.innerText.includes('Visual Monitoring Suite'));
                    if (visLabel) visLabel.click();
                })()
            """})
            await asyncio.sleep(2)

            # Click Mode B
            await call("Runtime.evaluate", {"expression": """
                (() => {
                    const labels = Array.from(document.querySelectorAll('label'));
                    const mB = labels.find(l => l.innerText && l.innerText.includes('Upload Your Own Satellite Imagery'));
                    if (mB) mB.click();
                })()
            """})
            await asyncio.sleep(2)

            # Upload Rhone file
            doc = await call("DOM.getDocument")
            root_id = doc["result"]["root"]["nodeId"]
            q = await call("DOM.querySelector", {"nodeId": root_id, "selector": 'input[type="file"]'})
            file_id = q["result"]["nodeId"]
            rhone_path = r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\datasets\objective4\sentinel2_rhone\rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif"
            await call("DOM.setFileInputFiles", {"files": [rhone_path], "nodeId": file_id})
            await asyncio.sleep(4)

            # Click NDSI tab
            await call("Runtime.evaluate", {"expression": """
                (() => {
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    const nTab = tabs.find(t => t.innerText && t.innerText.includes('NDSI Snow/Ice Index'));
                    if (nTab) nTab.click();
                })()
            """})
            await asyncio.sleep(3)

            res = await call("Runtime.evaluate", {"expression": "document.body.innerText"})
            text = res.get("result", {}).get("result", {}).get("value", "")
            print("--- TEXT IN PAGE ---")
            for line in text.splitlines():
                if any(k in line for k in ["NDSI", "Snow", "Pixels", "Ratio", "Coverage", "Mean", "24,459", "B3", "B11", "Heuristic"]):
                    print("  MATCH:", line)
    finally:
        proc.terminate()

if __name__ == "__main__":
    asyncio.run(check())
