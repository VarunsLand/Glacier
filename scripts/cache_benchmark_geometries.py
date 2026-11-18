"""Extract and pre-cache benchmark GLIMS geometries and metadata.
Produces datasets/glims/benchmark_glims_geometries.json for sub-millisecond
dynamic rendering in Objective 4.
"""
from __future__ import annotations

import json
from pathlib import Path
import pyogrio
import pyproj
import shapely
from shapely.ops import transform, unary_union
from shapely.wkt import dumps as to_wkt

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"
OUT_JSON = ROOT / "datasets" / "glims" / "benchmark_glims_geometries.json"

TARGETS = [
    {
        "id": "G008379E46648N",
        "key": "rhone",
        "name": "Rhone Glacier (Rhonegletscher)",
        "region": "Valais Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "figure_name": "glims_boundary_rhone.png",
    },
    {
        "id": "G008032E46504N",
        "key": "great_aletsch",
        "name": "Great Aletsch Glacier (Grosser Aletschgletscher)",
        "region": "Bernese Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "figure_name": "glims_boundary_great_aletsch.png",
    },
    {
        "id": "G008187E46569N",
        "key": "unteraar",
        "name": "Unteraar Glacier (Unteraargletscher)",
        "region": "Bernese Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "figure_name": "glims_boundary_unteraar.png",
    },
    {
        "id": "G007800E45965N",
        "key": "gorner",
        "name": "Gorner Glacier (Gornergletscher)",
        "region": "Pennine Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "figure_name": "glims_boundary_gorner.png",
    },
    {
        "id": "G007880E45990N",
        "key": "findel",
        "name": "Findel Glacier (Findelgletscher)",
        "region": "Pennine Alps, Switzerland",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
        "crs": "EPSG:32632",
        "figure_name": "glims_boundary_findel.png",
    },
]


def extract():
    print("Extracting benchmark geometries from GLIMS shapefile...")
    results = {}
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform

    for target in TARGETS:
        gid = target["id"]
        print(f"Reading {target['name']} ({gid})...")
        where_clause = f"glac_id = '{gid}' AND line_type = 'glac_bound'"
        gdf = pyogrio.read_dataframe(SHP_PATH, where=where_clause)
        gdf["_date"] = gdf["src_date"].str[:10]

        e_df = gdf[gdf["_date"] == target["t1_date"]]
        l_df = gdf[gdf["_date"] == target["t2_date"]]

        e_anlys = int(e_df["anlys_id"].iloc[0]) if len(e_df) > 0 and "anlys_id" in e_df.columns else 342911
        l_anlys = int(l_df["anlys_id"].iloc[0]) if len(l_df) > 0 and "anlys_id" in l_df.columns else 1095967

        geom_e = unary_union(e_df.geometry.values)
        geom_l = unary_union(l_df.geometry.values)

        geom_e_utm = transform(project, geom_e)
        geom_l_utm = transform(project, geom_l)

        area_e = geom_e_utm.area / 1e6
        area_l = geom_l_utm.area / 1e6
        retained = geom_e_utm.intersection(geom_l_utm)
        lost = geom_e_utm.difference(geom_l_utm)
        gained = geom_l_utm.difference(geom_e_utm)

        area_ret = retained.area / 1e6
        area_lost = lost.area / 1e6
        area_gain = gained.area / 1e6
        net_loss = area_e - area_l
        pct_loss = (net_loss / area_e) * 100.0

        results[target["key"]] = {
            "id": gid,
            "key": target["key"],
            "name": target["name"],
            "region": target["region"],
            "crs": target["crs"],
            "earlier_date": target["t1_date"],
            "later_date": target["t2_date"],
            "earlier_analysis_id": e_anlys,
            "later_analysis_id": l_anlys,
            "earlier_area_km2": round(area_e, 3),
            "later_area_km2": round(area_l, 3),
            "net_loss_km2": round(net_loss, 3),
            "percentage_change": round(pct_loss, 2),
            "retained_ice_area_km2": round(area_ret, 3),
            "lost_ice_area_km2": round(area_lost, 3),
            "gained_ice_area_km2": round(area_gain, 3),
            "figure_name": target["figure_name"],
            "bounds": list(geom_e_utm.bounds),
            "wkt_earlier": to_wkt(geom_e_utm),
            "wkt_later": to_wkt(geom_l_utm),
            "wkt_retained": to_wkt(retained),
            "wkt_lost": to_wkt(lost),
            "wkt_gained": to_wkt(gained) if not gained.is_empty else None,
        }
        print(f"  Done: {target['name']}: {area_e:.3f} -> {area_l:.3f} km2 (Net: -{net_loss:.3f} km2)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully cached benchmark geometries to {OUT_JSON}")


if __name__ == "__main__":
    extract()
