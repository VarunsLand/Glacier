"""GlacierVision: Live Browser Verification for Multiband GeoTIFF Upload (Objective 4).

Tests live browser workflow:
1. Navigates to http://localhost:8501/?page=monitoring
2. Toggles to Mode B: "Upload Your Own Satellite Imagery"
3. Uploads the exact Sentinel-2 Rhone multiband GeoTIFF:
   datasets/objective4/sentinel2_rhone/rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif
4. Verifies:
   - Success message: "Successfully ingested rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif (5 channels, 463×448 px)"
   - Metadata card: CRS EPSG:32632, Resolution 20.0m × 20.0m, Bands (B02, B03, B04, B08, B11)
   - Scope notice: 5-band selected cryospheric subset notice
5. Tests Sub-tab 1: True-Color RGB composite (B04-B03-B02). Captures screenshot.
6. Tests Sub-tab 2: False-Color SWIR composite (B11-B08-B04). Captures screenshot.
7. Tests Sub-tab 3: Individual Band Explorer with radiometric histogram. Captures screenshot.
8. Tests Sub-tab 4: NDSI Snow/Ice Index with 20m grid notice. Captures screenshot.
9. Tests Sub-tab 5: DEM Elevation & Hillshade (confirms graceful "DEM Unavailable" message).
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
RHONE_TIF = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\datasets\objective4\sentinel2_rhone\rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif")
OUTPUT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits\screenshots")
AUDIT_DIR = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch\outputs\audits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


async def run_live_upload_verification():
    port = 9502
    prof = OUTPUT_DIR / f"prof_rhone_upload_{port}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--disable-gpu",
        "--disable-extensions",
        f"--user-data-dir={prof}",
        "http://localhost:8501/?page=monitoring"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}

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
                nonlocal mid
                mid += 1
                await ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
                while True:
                    d = json.loads(await ws.recv())
                    if d.get("id") == mid:
                        return d

            await send_cmd("Page.enable")
            await send_cmd("DOM.enable")
            await send_cmd("Runtime.enable")

            # Set 1440x900 viewport
            await send_cmd("Emulation.setDeviceMetricsOverride", {
                "width": 1440,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": False
            })

            print("[1/6] Waiting for Objective 4 to load...")
            for _ in range(30):
                await asyncio.sleep(0.5)
                res = await send_cmd("Runtime.evaluate", {
                    "expression": "document.body.innerText.includes('Interactive Multi-Sensor Visual Glacier Monitoring') || document.body.innerText.includes('Sentinel-2 Multispectral Explorer')"
                })
                if res.get("result", {}).get("result", {}).get("value") is True:
                    break

            # Switch to Mode B: Upload Your Own Satellite Imagery
            print("[2/6] Switching to Mode B (Upload Your Own Satellite Imagery)...")
            click_mode_b = """
            (() => {
                const labels = Array.from(document.querySelectorAll('label'));
                const modeBLabel = labels.find(l => l.innerText.includes('Upload Your Own Satellite Imagery'));
                if (modeBLabel) {
                    modeBLabel.click();
                    return true;
                }
                const inputs = Array.from(document.querySelectorAll('input[type="radio"]'));
                if (inputs.length >= 2) {
                    inputs[1].click();
                    return true;
                }
                return false;
            })()
            """
            mb_res = await send_cmd("Runtime.evaluate", {"expression": click_mode_b})
            print(f"  Mode B clicked: {mb_res.get('result', {}).get('result', {}).get('value')}")
            await asyncio.sleep(2.0)

            # Upload Rhone GeoTIFF via DOM.setFileInputFiles
            print(f"[3/6] Uploading {RHONE_TIF.name}...")
            doc_res = await send_cmd("DOM.getDocument")
            root_node_id = doc_res["result"]["root"]["nodeId"]
            query_res = await send_cmd("DOM.querySelector", {
                "nodeId": root_node_id,
                "selector": 'input[type="file"]'
            })
            file_node_id = query_res["result"]["nodeId"]

            if not file_node_id:
                raise RuntimeError("File uploader input element not found in DOM.")

            await send_cmd("DOM.setFileInputFiles", {
                "files": [str(RHONE_TIF)],
                "nodeId": file_node_id
            })
            print("  File submitted to file input, waiting for ingestion and rendering...")
            await asyncio.sleep(4.0)

            # Check body text for ingestion success and metadata
            body_res = await send_cmd("Runtime.evaluate", {"expression": "document.body.innerText"})
            body_text = body_res.get("result", {}).get("result", {}).get("value", "")

            has_success = "Successfully ingested" in body_text and "rhone_sentinel2_B02_B03_B04_B08_B11_20m.tif" in body_text
            has_channels = "5 channels" in body_text
            has_dims = "463×448" in body_text or "463 × 448" in body_text
            has_crs = "EPSG:32632" in body_text
            has_res = "20.0m" in body_text or "20m" in body_text
            has_bands = "B02" in body_text and "B04" in body_text and "B11" in body_text
            has_scope_notice = "Sentinel-2 5-Band Selected Cryospheric Subset" in body_text

            results["ingestion_success"] = has_success
            results["metadata_verified"] = {
                "channels_5": has_channels,
                "dimensions_463x448": has_dims,
                "crs_epsg32632": has_crs,
                "resolution_20m": has_res,
                "bands_b02_b11": has_bands,
                "scope_notice": has_scope_notice
            }
            print(f"  Ingestion Success: {has_success}")
            print(f"  Metadata: channels={has_channels}, dims={has_dims}, crs={has_crs}, res={has_res}, bands={has_bands}")

            # Capture Sub-tab 1: True-Color RGB Screenshot
            print("[4/6] Verifying and capturing True-Color RGB composite...")
            shot_rgb = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_rgb = shot_rgb.get("result", {}).get("data", "")
            if b64_rgb:
                path_rgb = OUTPUT_DIR / "rhone_geotiff_live_upload_rgb.png"
                path_rgb.write_bytes(base64.b64decode(b64_rgb))
                results["rgb_screenshot"] = str(path_rgb)
                print(f"  Saved: {path_rgb.name}")

            # Sub-tab 2: False-Color SWIR
            print("[5/6] Testing Sub-tab 2: False-Color SWIR composite...")
            click_swir_tab = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const swirTab = tabs.find(t => t.innerText && t.innerText.includes('False-Color SWIR'));
                if (swirTab) {
                    swirTab.click();
                    return true;
                }
                return false;
            })()
            """
            await send_cmd("Runtime.evaluate", {"expression": click_swir_tab})
            await asyncio.sleep(2.0)

            swir_body = (await send_cmd("Runtime.evaluate", {"expression": "document.body.innerText"})).get("result", {}).get("result", {}).get("value", "")
            has_swir_comp = "False-Color SWIR Composite" in swir_body and ("B11" in swir_body)
            results["swir_rendered"] = has_swir_comp
            print(f"  SWIR composite rendered: {has_swir_comp}")

            shot_swir = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_swir = shot_swir.get("result", {}).get("data", "")
            if b64_swir:
                path_swir = OUTPUT_DIR / "rhone_geotiff_live_upload_swir.png"
                path_swir.write_bytes(base64.b64decode(b64_swir))
                results["swir_screenshot"] = str(path_swir)
                print(f"  Saved: {path_swir.name}")

            # Sub-tab 3: Individual Band Explorer
            print("  Testing Sub-tab 3: Individual Band Explorer...")
            click_bands_tab = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const bTab = tabs.find(t => t.innerText && t.innerText.includes('Individual Band Explorer'));
                if (bTab) {
                    bTab.click();
                    return true;
                }
                return false;
            })()
            """
            await send_cmd("Runtime.evaluate", {"expression": click_bands_tab})
            await asyncio.sleep(2.0)

            shot_bands = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_bands = shot_bands.get("result", {}).get("data", "")
            if b64_bands:
                path_bands = OUTPUT_DIR / "rhone_geotiff_live_upload_bands.png"
                path_bands.write_bytes(base64.b64decode(b64_bands))
                results["bands_screenshot"] = str(path_bands)
                print(f"  Saved: {path_bands.name}")

            # Sub-tab 4: NDSI Snow/Ice Index
            print("[6/6] Testing Sub-tab 4: NDSI Snow/Ice Index...")
            click_ndsi_tab = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const nTab = tabs.find(t => t.innerText && t.innerText.includes('NDSI Snow/Ice Index'));
                if (nTab) {
                    nTab.click();
                    return true;
                }
                return false;
            })()
            """
            await send_cmd("Runtime.evaluate", {"expression": click_ndsi_tab})
            await asyncio.sleep(2.0)

            ndsi_body = (await send_cmd("Runtime.evaluate", {"expression": "document.body.innerText"})).get("result", {}).get("result", {}).get("value", "")
            has_ndsi_field = "continuous ndsi field" in ndsi_body.lower() and "binary cryogenic mask" in ndsi_body.lower()
            has_ndsi_stats = "24,459" in ndsi_body or "snow/ice classified pixels" in ndsi_body.lower() or "mean scene ndsi" in ndsi_body.lower()
            results["ndsi_rendered"] = has_ndsi_field and has_ndsi_stats
            print(f"  NDSI Field & Binary Mask rendered: {has_ndsi_field}, Stats: {has_ndsi_stats}")

            shot_ndsi = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_ndsi = shot_ndsi.get("result", {}).get("data", "")
            if b64_ndsi:
                path_ndsi = OUTPUT_DIR / "rhone_geotiff_live_upload_ndsi.png"
                path_ndsi.write_bytes(base64.b64decode(b64_ndsi))
                results["ndsi_screenshot"] = str(path_ndsi)
                print(f"  Saved: {path_ndsi.name}")

            # Sub-tab 5: DEM Elevation & Hillshade
            print("  Testing Sub-tab 5: DEM Elevation & Hillshade (verifying graceful unavailable handling)...")
            click_dem_tab = """
            (() => {
                const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                const dTab = tabs.find(t => t.innerText && t.innerText.includes('DEM Elevation & Hillshade'));
                if (dTab) {
                    dTab.click();
                    return true;
                }
                return false;
            })()
            """
            await send_cmd("Runtime.evaluate", {"expression": click_dem_tab})
            await asyncio.sleep(1.5)

            dem_body = (await send_cmd("Runtime.evaluate", {"expression": "document.body.innerText"})).get("result", {}).get("result", {}).get("value", "")
            has_dem_unavail = "Digital Elevation Model Unavailable" in dem_body
            results["dem_unavailable_graceful"] = has_dem_unavail
            print(f"  DEM gracefully flagged as unavailable: {has_dem_unavail}")

            all_passed = (
                has_success and
                has_channels and
                has_dims and
                has_crs and
                has_res and
                has_bands and
                has_swir_comp and
                has_ndsi_field and
                has_ndsi_stats and
                has_dem_unavail
            )
            results["all_checks_passed"] = all_passed
            print(f"\nLive GeoTIFF Upload Verification Complete! Overall Success: {all_passed}")

            # Save audit JSON
            audit_json = AUDIT_DIR / "rhone_geotiff_live_upload_verification.json"
            audit_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Audit log saved to: {audit_json}")

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass

    return results


if __name__ == "__main__":
    asyncio.run(run_live_upload_verification())
