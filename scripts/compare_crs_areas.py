"""GlacierVision: Comprehensive CRS Projection Comparison for Benchmark Glaciers.

Compares geometric area metrics across:
1. EPSG:3035 (ETRS89-extended / LAEA Europe - Lambert Azimuthal Equal Area, INSPIRE / EEA standard)
2. EPSG:32632 (WGS 84 / UTM Zone 32N - Conformal Transverse Mercator, k0 = 0.9996)
3. EPSG:2056 (CH1903+ / LV95 - Swiss National Grid, swisstopo authoritative survey)
4. Geodesic / Ellipsoidal WGS84 area via PyProj Geod (true ellipsoidal area)

For all 3 benchmark glaciers:
- Rhone Glacier (G008379E46648N)
- Great Aletsch Glacier (G008032E46504N)
- Unteraar Glacier (G008187E46569N)
- Plus Gornergletscher (G007800E45965N)
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pyogrio
import pyproj
from shapely.ops import transform, unary_union

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
SHP_PATH = ROOT / "datasets" / "glims" / "extracted" / "glims_download_52211" / "glims_polygons.shp"

BENCHMARKS = [
    {
        "id": "G008379E46648N",
        "name": "Rhone Glacier",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
    },
    {
        "id": "G008032E46504N",
        "name": "Great Aletsch Glacier",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
    },
    {
        "id": "G008187E46569N",
        "name": "Unteraar Glacier",
        "t1_date": "1973-09-01",
        "t2_date": "2015-08-29",
    },
]

CRS_LIST = [
    ("EPSG:3035", "ETRS89 / LAEA Europe (Lambert Azimuthal Equal Area - INSPIRE / EEA standard)"),
    ("EPSG:32632", "WGS 84 / UTM Zone 32N (Transverse Mercator - Regional glaciology baseline)"),
    ("EPSG:2056", "CH1903+ / LV95 (Swiss National Grid - swisstopo official)"),
]

def analyze_crs():
    results = {}
    where_clause = "glac_id IN ('G008379E46648N', 'G008032E46504N', 'G008187E46569N') AND line_type = 'glac_bound'"
    print("Loading benchmark polygons from shapefile...")
    gdf = pyogrio.read_dataframe(SHP_PATH, where=where_clause)
    gdf["_date"] = gdf["src_date"].str[:10]

    for b in BENCHMARKS:
        gid = b["id"]
        name = b["name"]
        print(f"\n{'='*75}\nANALYSIS FOR {name.upper()} ({gid})\n{'='*75}")
        
        b_gdf = gdf[gdf["glacier_id"] == gid] if "glacier_id" in gdf.columns else gdf[gdf["glac_id"] == gid]
        e_df = b_gdf[b_gdf["_date"] == b["t1_date"]]
        l_df = b_gdf[b_gdf["_date"] == b["t2_date"]]
        
        geom_e_4326 = unary_union(e_df.geometry.values)
        geom_l_4326 = unary_union(l_df.geometry.values)

        # Geodetic ellipsoidal area
        geod = pyproj.Geod(ellps="WGS84")
        area_e_geod = abs(geod.geometry_area_perimeter(geom_e_4326)[0]) / 1e6
        area_l_geod = abs(geod.geometry_area_perimeter(geom_l_4326)[0]) / 1e6
        net_loss_geod = area_e_geod - area_l_geod
        pct_loss_geod = (net_loss_geod / area_e_geod) * 100.0

        results[gid] = {
            "name": name,
            "geodetic_wgs84": {
                "earlier_km2": area_e_geod,
                "later_km2": area_l_geod,
                "net_loss_km2": net_loss_geod,
                "pct_loss": pct_loss_geod,
            },
            "crs_metrics": {},
        }

        print(f"True Geodesic Ellipsoidal Area (WGS84): Earlier={area_e_geod:.6f} km², Modern={area_l_geod:.6f} km², Net={net_loss_geod:.6f} km² ({pct_loss_geod:.4f}%)")

        for crs_code, crs_desc in CRS_LIST:
            proj = pyproj.Transformer.from_crs("EPSG:4326", crs_code, always_xy=True).transform
            geom_e = transform(proj, geom_e_4326)
            geom_l = transform(proj, geom_l_4326)

            ae = geom_e.area / 1e6
            al = geom_l.area / 1e6
            retained = geom_e.intersection(geom_l)
            lost = geom_e.difference(geom_l)
            gained = geom_l.difference(geom_e)

            a_ret = retained.area / 1e6
            a_lost = lost.area / 1e6
            a_gain = gained.area / 1e6
            net_loss = a_lost - a_gain
            pct_loss = (net_loss / ae) * 100.0
            ann_rate = net_loss / 42.0

            # Identity residuals
            id1 = abs(ae - (a_ret + a_lost))
            id2 = abs(al - (a_ret + a_gain))
            id3 = abs(net_loss - (ae - al))

            results[gid]["crs_metrics"][crs_code] = {
                "description": crs_desc,
                "earlier_km2": ae,
                "later_km2": al,
                "retained_km2": a_ret,
                "lost_km2": a_lost,
                "gained_km2": a_gain,
                "net_loss_km2": net_loss,
                "pct_loss": pct_loss,
                "annual_rate_km2_yr": ann_rate,
                "id1_res": id1,
                "id2_res": id2,
                "id3_res": id3,
                "diff_vs_geodetic_pct": ((ae - area_e_geod) / area_e_geod) * 100.0,
            }

            print(f"\n--- {crs_code} : {crs_desc[:40]}... ---")
            print(f"  Earlier Area:  {ae:.6f} km² (diff vs geodetic: {((ae - area_e_geod)/area_e_geod)*100:+.4f}%)")
            print(f"  Modern Area:   {al:.6f} km² (diff vs geodetic: {((al - area_l_geod)/area_l_geod)*100:+.4f}%)")
            print(f"  Retained Ice:  {a_ret:.6f} km²")
            print(f"  Lost Area:     {a_lost:.6f} km²")
            print(f"  Gained Area:   {a_gain:.6f} km²")
            print(f"  Net Loss:      {net_loss:.6f} km² | Shrinkage: {pct_loss:.4f}% | Annual: {ann_rate:.4f} km²/yr")
            print(f"  Identities:    id1={id1:.2e}, id2={id2:.2e}, id3={id3:.2e}")

    out_p = ROOT / "scratch" / "crs_comparison_results.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CRS comparison results to {out_p}")

if __name__ == "__main__":
    analyze_crs()
