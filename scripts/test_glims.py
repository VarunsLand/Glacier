import requests
import xml.etree.ElementTree as ET

BASE = "https://www.glims.org/geoserver/ows"

for service, version in [("WMS", "1.3.0"), ("WFS", "2.0.0")]:

    print("\n==========", service, "TEST ==========")

    try:
        r = requests.get(
            BASE,
            params={
                "service": service,
                "version": version,
                "request": "GetCapabilities"
            },
            timeout=90
        )

        print("HTTP:", r.status_code)
        print("Content type:", r.headers.get("Content-Type"))
        print("Response:", r.text[:300])

        r.raise_for_status()
        xml = ET.fromstring(r.content)

        if service == "WFS":
            for item in xml.findall(".//{*}FeatureType"):
                name = item.find("{*}Name")
                title = item.find("{*}Title")
                if name is not None:
                    print(
                        "LAYER:", name.text,
                        "| TITLE:", title.text if title is not None else ""
                    )

        else:
            for item in xml.findall(".//{*}Layer"):
                name = item.find("{*}Name")
                if name is not None:
                    print("LAYER:", name.text)

    except Exception as e:
        print("ERROR:", repr(e))
