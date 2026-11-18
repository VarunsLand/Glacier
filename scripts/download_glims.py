import requests, json, time
from pathlib import Path

OUT = Path.home() / "Desktop" / "GlacierResearch" / "datasets" / "glims"
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://www.glims.org/geoserver/GLIMS/ows"
LAYER = "GLIMS:GLIMS_Glacier_Outlines"
PAGE = 1000
offset = 0
total = 0
seen_pages = set()

session = requests.Session()
session.headers["User-Agent"] = "GlacierResearch/1.0"

print("Starting global GLIMS WFS export...")

while True:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "outputFormat": "application/json",
        "count": PAGE,
        "startIndex": offset
    }

    response = session.get(URL, params=params, timeout=180)
    response.raise_for_status()

    try:
        data = response.json()
    except Exception:
        raise RuntimeError("GLIMS returned non-JSON data: " + response.text[:1000])

    features = data.get("features", [])

    if not features:
        break

    signature = tuple(
        str(f.get("id", "")) for f in features[:10]
    )

    if signature in seen_pages:
        raise RuntimeError(
            "Repeated WFS page detected. Stopping to prevent duplicate export."
        )

    seen_pages.add(signature)

    filename = OUT / f"glims_{offset:09d}.geojson"
    filename.write_text(json.dumps(data), encoding="utf-8")

    total += len(features)
    print(f"Saved {total:,} features | latest page: {filename.name}")

    offset += len(features)

    if len(features) < PAGE:
        break

    time.sleep(0.5)

metadata = {
    "source": "GLIMS",
    "service": URL,
    "layer": LAYER,
    "features_downloaded": total,
    "pages_downloaded": len(seen_pages)
}

(OUT / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nGLIMS EXPORT FINISHED")
print("Total features:", total)
print("Saved at:", OUT)
