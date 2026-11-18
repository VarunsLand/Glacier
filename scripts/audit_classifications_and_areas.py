"""GlacierVision: Detailed Audit of Classification Counts and Area Calculations.
"""
import pandas as pd
import numpy as np

csv_path = "C:/Users/Varun's LOQ/Desktop/GlacierResearch/outputs/results/glims_retreat_assessment_validated.csv"
df = pd.read_csv(csv_path)

print(f"Total rows in CSV: {len(df):,}")
print(f"Unique glacier IDs: {df['glacier_id'].nunique():,}")
assert len(df) == df['glacier_id'].nunique(), "Duplicate glacier IDs found!"

# Check zero-threshold partition:
zero_ret = (df['area_loss_km2'] > 0).sum()
zero_adv = (df['area_loss_km2'] < 0).sum()
zero_sta = (df['area_loss_km2'] == 0).sum()
print("\n--- ZERO-THRESHOLD PARTITION (area_loss_km2 > 0 vs == 0 vs < 0) ---")
print(f"Strict Loss (>0):   {zero_ret:,} ({zero_ret/len(df)*100:.2f}%)")
print(f"Strict Gain (<0):   {zero_adv:,} ({zero_adv/len(df)*100:.2f}%)")
print(f"Strict Zero (==0):  {zero_sta:,} ({zero_sta/len(df)*100:.2f}%)")
print(f"Sum:                {zero_ret + zero_adv + zero_sta:,} ({(zero_ret+zero_adv+zero_sta)/len(df)*100:.2f}%)")

# Check glaciological threshold partition (2% deadband):
dyn_counts = df['dynamics_class'].value_counts()
print("\n--- GLACIOLOGICAL DEADBAND PARTITION (dynamics_class) ---")
for k, v in dyn_counts.items():
    print(f"{k:22s}: {v:,} ({v/len(df)*100:.2f}%)")
print(f"Total:                 {dyn_counts.sum():,} ({dyn_counts.sum()/len(df)*100:.2f}%)")

# Grouped glaciological 3-class partition:
sig_ret = dyn_counts.get("Significant Retreat", 0)
mod_ret = dyn_counts.get("Moderate Retreat", 0)
tot_ret = sig_ret + mod_ret
adv_grow = dyn_counts.get("Advance / Growth", 0)
sta_equil = dyn_counts.get("Stable / Equilibrium", 0)

print("\n--- MUTUALLY EXCLUSIVE 3-CLASS GLACIOLOGICAL PARTITION ---")
print(f"Retreating (>= 2%):    {tot_ret:,} ({tot_ret/len(df)*100:.2f}%)")
print(f"Advancing (<= -2%):    {adv_grow:,} ({adv_grow/len(df)*100:.2f}%)")
print(f"Stable (-2% to +2%):   {sta_equil:,} ({sta_equil/len(df)*100:.2f}%)")
print(f"Sum:                   {tot_ret + adv_grow + sta_equil:,} ({(tot_ret + adv_grow + sta_equil)/len(df)*100:.2f}%)")

# Area Verification:
a1_sum = df['earlier_area_km2'].sum()
a2_sum = df['later_area_km2'].sum()
gross_lost = df.loc[df['area_loss_km2'] > 0, 'area_loss_km2'].sum()
gross_gained = abs(df.loc[df['area_loss_km2'] < 0, 'area_loss_km2'].sum())
net_loss = df['area_loss_km2'].sum()

print("\n--- AREA MASS BALANCE VERIFICATION ---")
print(f"Earlier Total Area:   {a1_sum:,.2f} km^2")
print(f"Later Total Area:     {a2_sum:,.2f} km^2")
print(f"Gross Area Lost:      {gross_lost:,.2f} km^2")
print(f"Gross Area Gained:    {gross_gained:,.2f} km^2")
print(f"Gross Lost - Gained:  {gross_lost - gross_gained:,.2f} km^2")
print(f"Net Area Loss (Sum):  {net_loss:,.2f} km^2")
print(f"Earlier - Later:      {a1_sum - a2_sum:,.2f} km^2")
print(f"Diff:                 {abs((gross_lost - gross_gained) - net_loss):.6f}")

agg_pct = ((a1_sum - a2_sum) / a1_sum) * 100.0
mean_pct = df['percentage_change'].mean()
median_pct = df['percentage_change'].median()
print(f"Aggregate Pct Loss:   {agg_pct:.2f}%")
print(f"Mean Indiv Pct:       {mean_pct:.2f}%")
print(f"Median Indiv Pct:     {median_pct:.2f}%")
