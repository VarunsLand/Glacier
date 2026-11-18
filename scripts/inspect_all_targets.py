import json

data = json.load(open('outputs/results/target_glaciers_metadata.json'))
for gid in ['G008379E46648N', 'G008032E46504N', 'G008187E46569N', 'G007800E45965N', 'G007880E45990N']:
    print(f"=== {gid} ===")
    for r in data.get(gid, []):
        dt = r['src_date'][:10]
        if dt.startswith("1973") or dt.startswith("2015") or dt.startswith("2016") or dt.startswith("2010"):
            print(f"  anlys_id={r['anlys_id']}, date={dt}, name={r['glac_name']}, db_area={r['db_area']}")
