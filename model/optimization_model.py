"""
optimization_model.py
---------------------
Builds and solves a linear-programming model for optimal crop–land allocation
using the PuLP library.

Objective:   Maximise total food production (tonnes).
Constraints: Land, water, food-demand minimum, and land-degradation limit.
"""

import os
import json
import pandas as pd
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ===================================================================
#  1. Load model inputs
# ===================================================================

def load_model_data(model_input_path: str = None,
                    constraints_path: str = None):
    """
    Load the preprocessed model-input CSV and system-constraints CSV.

    Returns
    -------
    crops_df      : pd.DataFrame   per-crop parameters
    constraints   : dict           land_available, water_available, food_demand
    """
    if model_input_path is None:
        model_input_path = os.path.join(DATA_DIR, "model_input.csv")
    if constraints_path is None:
        constraints_path = os.path.join(DATA_DIR, "system_constraints.csv")

    crops_df = pd.read_csv(model_input_path)
    constr_df = pd.read_csv(constraints_path)

    constraints = {
        "land_available":  constr_df["land_available"].iloc[0],
        "water_available": constr_df["water_available"].iloc[0],
        "food_demand":     constr_df["food_demand"].iloc[0],
    }

    return crops_df, constraints


# ===================================================================
#  2. Build the LP model
# ===================================================================

def build_model(crops_df: pd.DataFrame,
                constraints: dict,
                degradation_limit: float = 1.0):
    """
    Create the PuLP LP model.

    Parameters
    ----------
    crops_df : DataFrame with columns [crop, yield, water_requirement, degradation_index]
    constraints : dict with keys land_available, water_available, food_demand
    degradation_limit : upper bound on the sum of (degradation_index × area)

    Returns
    -------
    prob : LpProblem
    x    : dict[str, LpVariable]  – decision variables keyed by crop name
    """

    prob = LpProblem("WEFL_Crop_Land_Allocation", LpMaximize)

    # --- Decision variables: X_crop ≥ 0 (hectares) ---
    crop_names = crops_df["crop"].tolist()
    x = {
        crop: LpVariable(f"X_{crop.replace(' ', '_').replace(',', '')}",
                          lowBound=0, cat="Continuous")
        for crop in crop_names
    }

    # Look-up helpers
    yield_map   = dict(zip(crops_df["crop"], crops_df["yield"]))
    water_map   = dict(zip(crops_df["crop"], crops_df["water_requirement"]))
    degrad_map  = dict(zip(crops_df["crop"], crops_df["degradation_index"]))

    # --- Objective: Maximise Σ (yield_crop × X_crop) ---
    prob += (
        lpSum(yield_map[c] * x[c] for c in crop_names),
        "Maximise_Total_Food_Production",
    )

    # --- Constraint 1: Land ---
    prob += (
        lpSum(x[c] for c in crop_names) <= constraints["land_available"],
        "Land_Constraint",
    )

    # --- Constraint 2: Water ---
    prob += (
        lpSum(water_map[c] * x[c] for c in crop_names) <= constraints["water_available"],
        "Water_Constraint",
    )

    # --- Constraint 3: Food demand (minimum production) ---
    prob += (
        lpSum(yield_map[c] * x[c] for c in crop_names) >= constraints["food_demand"],
        "Food_Demand_Constraint",
    )

    # --- Constraint 4: Degradation ---
    prob += (
        lpSum(degrad_map[c] * x[c] for c in crop_names) <= degradation_limit * constraints["land_available"],
        "Degradation_Constraint",
    )

    return prob, x


# ===================================================================
#  3. Solve
# ===================================================================

def solve_model(prob, x, crops_df):
    """
    Solve the LP and return results.

    Returns
    -------
    allocation : pd.DataFrame   – [crop, optimal_area]
    metrics    : dict            – summary statistics
    status     : str             – solver status
    """
    prob.solve()  # default CBC solver
    status = LpStatus[prob.status]
    print(f"\n  Solver status: {status}")

    if status != "Optimal":
        print("  [WARNING] No optimal solution found.")
        return None, None, status

    # Build allocation table
    yield_map  = dict(zip(crops_df["crop"], crops_df["yield"]))
    water_map  = dict(zip(crops_df["crop"], crops_df["water_requirement"]))
    degrad_map = dict(zip(crops_df["crop"], crops_df["degradation_index"]))

    records = []
    for crop in crops_df["crop"]:
        area = value(x[crop])
        records.append({
            "crop": crop,
            "optimal_area": round(area, 2),
        })

    allocation = pd.DataFrame(records)

    # Metrics
    total_production = sum(yield_map[c] * value(x[c]) for c in crops_df["crop"])
    total_water      = sum(water_map[c] * value(x[c]) for c in crops_df["crop"])
    total_land       = sum(value(x[c])                  for c in crops_df["crop"])
    degradation_score = sum(degrad_map[c] * value(x[c]) for c in crops_df["crop"])

    metrics = {
        "total_production":  round(total_production, 2),
        "total_water_used":  round(total_water, 2),
        "total_land_used":   round(total_land, 2),
        "degradation_score": round(degradation_score, 4),
    }

    return allocation, metrics, status


# ===================================================================
#  4. Export results
# ===================================================================

def export_results(allocation: pd.DataFrame, metrics: dict):
    """Save allocation CSV and metrics JSON to results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    alloc_path  = os.path.join(RESULTS_DIR, "optimal_land_allocation.csv")
    metric_path = os.path.join(RESULTS_DIR, "model_metrics.json")

    allocation.to_csv(alloc_path, index=False)
    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [OK] Saved {alloc_path}")
    print(f"  [OK] Saved {metric_path}")


# ===================================================================
#  5. Convenience runner
# ===================================================================

def run_optimization():
    """
    End-to-end: load data → build → solve → export.

    Returns
    -------
    allocation, metrics, status
    """
    print("\n========== OPTIMIZATION MODEL ==========\n")

    crops_df, constraints = load_model_data()
    print("  Crops in model  :", len(crops_df))
    print("  Land available  :", f"{constraints['land_available']:,.0f} ha")
    print("  Water available :", f"{constraints['water_available']:,.0f} m3")
    print("  Food demand     :", f"{constraints['food_demand']:,.2f}")

    prob, x = build_model(crops_df, constraints, degradation_limit=1.0)
    allocation, metrics, status = solve_model(prob, x, crops_df)

    if allocation is not None:
        export_results(allocation, metrics)

        # ---- Pretty-print summary ----
        print("\n---------- SUMMARY ----------")
        print(f"  Total production : {metrics['total_production']:,.2f} tonnes")
        print(f"  Total water used : {metrics['total_water_used']:,.0f} m3")
        print(f"  Total land used  : {metrics['total_land_used']:,.0f} ha")
        print(f"  Degradation score: {metrics['degradation_score']:.4f}")
        print("-----------------------------\n")

        # Show non-zero allocations
        nonzero = allocation[allocation["optimal_area"] > 0]
        if not nonzero.empty:
            print("  Crops with non-zero allocation:")
            for _, row in nonzero.iterrows():
                print(f"    {row['crop']:30s}  {row['optimal_area']:>14,.2f} ha")
        print()

    print("=========================================\n")
    return allocation, metrics, status


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_optimization()
