import pyogrio
import pyproj
from shapely.ops import transform, unary_union

shp = 'datasets/glims/extracted/glims_download_52211/glims_polygons.shp'
targets = {
    "Findel": "G007880E45990N",
    "Gorner": "G007800E45965N",
}

for name, gid in targets.items():
    print(f"\n==================== {name} ({gid}) ====================")
    df = pyogrio.read_dataframe(shp, where=f"glac_id = '{gid}' AND line_type = 'glac_bound'")
    df['_date'] = df['src_date'].str[:10]
    e_df = df[df['_date'] == '1973-09-01']
    l_df = df[df['_date'] == '2015-08-29']
    
    geom_e = unary_union(e_df.geometry.values)
    geom_l = unary_union(l_df.geometry.values)
    
    for crs_name in ['EPSG:3035', 'EPSG:32632', 'EPSG:2056']:
        proj = pyproj.Transformer.from_crs('EPSG:4326', crs_name, always_xy=True).transform
        ge = transform(proj, geom_e)
        gl = transform(proj, geom_l)
        
        ae = ge.area / 1e6
        al = gl.area / 1e6
        ret = ge.intersection(gl)
        lost = ge.difference(gl)
        gained = gl.difference(ge)
        
        a_ret = ret.area / 1e6
        a_lost = lost.area / 1e6
        a_gain = gained.area / 1e6
        net = ae - al
        pct = (net / ae) * 100.0
        
        print(f"--- {crs_name} ---")
        print(f"  Earlier Area: {ae:.4f} km²")
        print(f"  Modern Area:  {al:.4f} km²")
        print(f"  Retained:     {a_ret:.4f} km²")
        print(f"  Lost:         {a_lost:.4f} km²")
        print(f"  Gained:       {a_gain:.4f} km²")
        print(f"  Net Loss:     {net:.4f} km²")
        print(f"  Pct Change:   -{pct:.4f}%")
