import pyogrio
import pyproj
from shapely.ops import transform, unary_union
from pathlib import Path

shp = 'datasets/glims/extracted/glims_download_52211/glims_polygons.shp'
df = pyogrio.read_dataframe(shp, where="glac_id = 'G007880E45990N' AND line_type = 'glac_bound'")
df['_date'] = df['src_date'].str[:10]
e_df = df[df['_date'] == '1973-09-01']
l_df = df[df['_date'] == '2015-08-29']

print('--- DB_AREA Attributes ---')
db_e = e_df['db_area'].sum()
db_l = l_df['db_area'].sum()
print(f'Earlier db_area sum: {db_e:.6f} km² (rounded: {db_e:.3f})')
print(f'Later db_area sum:   {db_l:.6f} km² (rounded: {db_l:.3f})')
print(f'Net loss db_area:    {db_e - db_l:.6f} km² (rounded: {db_e - db_l:.3f})')

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
    
    print(f'\n--- {crs_name} ---')
    print(f'Earlier Area (At1):   {ae:.6f} km² (rounded: {ae:.3f})')
    print(f'Later Area (At2):     {al:.6f} km² (rounded: {al:.3f})')
    print(f'Retained Core Ice:    {a_ret:.6f} km² (rounded: {a_ret:.3f})')
    print(f'Lost Ice (Retreat):   {a_lost:.6f} km² (rounded: {a_lost:.3f})')
    print(f'Gained Ice (Advance): {a_gain:.6f} km² (rounded: {a_gain:.3f})')
    print(f'Net Loss (At1 - At2): {net:.6f} km² (rounded: {net:.3f})')
    print(f'Percentage Change:    -{pct:.4f}% (rounded: -{pct:.2f}%)')
    print(f'Identity 1 (Ret + Lost == At1): {a_ret + a_lost:.6f} vs {ae:.6f} (diff: {abs(a_ret + a_lost - ae):.2e})')
    print(f'Identity 2 (Ret + Gain == At2): {a_ret + a_gain:.6f} vs {al:.6f} (diff: {abs(a_ret + a_gain - al):.2e})')
    print(f'Identity 3 (Lost - Gain == Net): {a_lost - a_gain:.6f} vs {net:.6f} (diff: {abs(a_lost - a_gain - net):.2e})')
