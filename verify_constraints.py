"""Verify optimization constraints and print test summary."""
import json
import pandas as pd

mi = pd.read_csv("data/model_input.csv")
sc = pd.read_csv("data/system_constraints.csv")
alloc = pd.read_csv("results/optimal_land_allocation.csv")
with open("results/model_metrics.json") as f:
    metrics = json.load(f)

df = pd.merge(alloc, mi, on="crop")
land_avail = sc["land_available"].iloc[0]
water_avail = sc["water_available"].iloc[0]
food_demand = sc["food_demand"].iloc[0]
deg_limit = 1.0 * land_avail

total_land = df["optimal_area"].sum()
total_water = (df["optimal_area"] * df["water_requirement"]).sum()
total_prod = (df["optimal_area"] * df["yield"]).sum()
total_deg = (df["optimal_area"] * df["degradation_index"]).sum()

print("=" * 60)
print("  CONSTRAINT VERIFICATION")
print("=" * 60)

c1 = total_land <= land_avail
label1 = "PASS" if c1 else "FAIL"
print(f"  LAND:        {total_land:,.0f} <= {land_avail:,.0f}  => {label1}")

c2 = total_water <= water_avail
label2 = "PASS" if c2 else "FAIL"
print(f"  WATER:       {total_water:,.0f} <= {water_avail:,.0f}  => {label2}")

c3 = total_prod >= food_demand
label3 = "PASS" if c3 else "FAIL"
print(f"  FOOD DEMAND: {total_prod:,.2f} >= {food_demand:,.2f}  => {label3}")

c4 = total_deg <= deg_limit
label4 = "PASS" if c4 else "FAIL"
print(f"  DEGRADATION: {total_deg:,.0f} <= {deg_limit:,.0f}  => {label4}")

all_pass = all([c1, c2, c3, c4])
print()
print("=" * 60)
print("  TEST SUMMARY")
print("=" * 60)
print(f"  Number of crops optimized: {len(mi)}")
print(f"  Total land allocated:      {metrics['total_land_used']:,.0f} ha")
print(f"  Total water used:          {metrics['total_water_used']:,.0f} m3")
print(f"  Total production:          {metrics['total_production']:,.2f} tonnes")

status = "YES" if all_pass else "NO"
print(f"  All constraints satisfied: {status}")
print()
if all_pass:
    print("  Optimization completed successfully.")
else:
    print("  WARNING: Some constraints were not satisfied!")
print("=" * 60)

# Missing values check
print()
print("  MODEL INPUT DATA QUALITY:")
missing = mi.isnull().sum().sum()
print(f"  Missing values in model_input.csv: {missing}")
print(f"  Output files exist: optimal_land_allocation.csv, model_metrics.json")
print(f"  model_metrics.json fields: {list(metrics.keys())}")
print(f"  optimal_land_allocation.csv fields: {list(alloc.columns)}")
