import json
from pathlib import Path

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
p = ROOT / "scratch" / "benchmark_glims_metadata.json"
data = json.load(open(p, encoding="utf-8"))

by_gid = {}
for r in data:
    gid = r["glac_id"]
    by_gid.setdefault(gid, []).append(r)

for gid, rows in by_gid.items():
    print(f"\n=======================================================")
    print(f"GLACIER: {gid} (Total rows: {len(rows)})")
    print(f"=======================================================")
    dates = sorted(set(r["src_date"][:10] for r in rows))
    print(f"  Distinct src_dates: {dates}")
    for r in rows:
        if r["src_date"].startswith("1973"):
            print(f"  1973 record:")
            for k, v in sorted(r.items()):
                print(f"    {k}: {v}")
