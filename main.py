"""
main.py
-------
Entry point for the Water–Food–Land (WEFL) Optimization Model.

Workflow:
  1. Load and preprocess FAOSTAT datasets
  2. Build the linear-programming optimisation model
  3. Solve for optimal crop–land allocation
  4. Export results to results/
"""

import sys
import os

# Ensure the project root is on the path so imports work when running
# directly with `python main.py` from the wefl_model/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocess_data import run_preprocessing
from model.optimization_model import run_optimization


def main():
    print("=" * 55)
    print("  WATER - FOOD - LAND  OPTIMIZATION  MODEL  (WEFL)")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Step 1 & 2: Load datasets and run preprocessing
    # ------------------------------------------------------------------
    model_input, system_constraints = run_preprocessing()

    # ------------------------------------------------------------------
    # Step 3 & 4: Build optimisation model, solve, and export results
    # ------------------------------------------------------------------
    allocation, metrics, status = run_optimization()

    # ------------------------------------------------------------------
    # Step 5: Final summary
    # ------------------------------------------------------------------
    if status == "Optimal":
        print("=" * 55)
        print("  MODEL SOLVED SUCCESSFULLY")
        print("=" * 55)
        print(f"  Status           : {status}")
        print(f"  Total production : {metrics['total_production']:,.2f} tonnes")
        print(f"  Total water used : {metrics['total_water_used']:,.0f} m3")
        print(f"  Total land used  : {metrics['total_land_used']:,.0f} ha")
        print(f"  Degradation score: {metrics['degradation_score']:.4f}")
        print()
        print("  Results saved to:")
        print("    - results/optimal_land_allocation.csv")
        print("    - results/model_metrics.json")
        print("=" * 55)
    else:
        print(f"\n  [WARNING] Solver finished with status: {status}")
        print("  Check your data and constraints.\n")


if __name__ == "__main__":
    main()
