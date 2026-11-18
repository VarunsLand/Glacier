from src.inference import get_glacier_dynamic_assessment

tests = [
    'Rhone',
    'Great Aletsch',
    'Unteraar',
    'Gorner',
    'Findel',
    'G010725E46789N',
    'G057419E80930N',
    'INVALID_GLACIER_XYZ'
]

print(f"{'Input':<20} | {'Resolved ID':<15} | {'Glacier Name':<35} | {'Found':<5} | {'Earlier (km²)':<12} | {'Later (km²)':<12} | {'Net Loss':<10} | {'Pct Change'}")
print("-" * 135)
for t in tests:
    res = get_glacier_dynamic_assessment(t)
    e_anlys = res.get('earlier_analysis_id') or 'N/A'
    l_anlys = res.get('later_analysis_id') or 'N/A'
    e_area = f"{res['earlier_area_km2']:.3f}" if res['found'] else "N/A"
    l_area = f"{res['later_area_km2']:.3f}" if res['found'] else "N/A"
    loss = f"{res['area_change_km2']:+.3f}" if res['found'] else "N/A"
    pct = f"{res['percent_change']:+.2f}%" if res['found'] else "N/A"
    print(f"{t:<20} | {res['glacier_id']:<15} | {res['glacier_name'][:35]:<35} | {str(res['found']):<5} | {e_area:<12} | {l_area:<12} | {loss:<10} | {pct}")
