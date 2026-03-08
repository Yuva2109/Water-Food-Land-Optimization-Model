"""
preprocess_data.py
------------------
Loads raw FAOSTAT CSV datasets, cleans and standardises them, then produces
two derived files that feed directly into the optimisation model:

    data/model_input.csv      – per-crop yield, water requirement & degradation index
    data/system_constraints.csv – aggregate land, water & food-demand limits
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths  (resolved relative to this script's parent → wefl_model/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_CROP    = os.path.join(DATA_DIR, "crop_livestock.csv")
RAW_LAND    = os.path.join(DATA_DIR, "land_cover.csv")
RAW_FOOD    = os.path.join(DATA_DIR, "food_demand_processed.csv")
RAW_WATER   = os.path.join(DATA_DIR, "water_requirement.csv")

OUT_MODEL_INPUT       = os.path.join(DATA_DIR, "model_input.csv")
OUT_SYSTEM_CONSTRAINTS = os.path.join(DATA_DIR, "system_constraints.csv")


# ===================================================================
#  1. Loading helpers
# ===================================================================

def load_crop_data(filepath: str = RAW_CROP) -> pd.DataFrame:
    """Load FAOSTAT Crops & Livestock dataset."""
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  [load] crop_livestock.csv  -> {len(df):,} rows")
    return df


def load_land_data(filepath: str = RAW_LAND) -> pd.DataFrame:
    """Load FAOSTAT Land Cover dataset."""
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  [load] land_cover.csv      -> {len(df):,} rows")
    return df


def load_food_demand(filepath: str = RAW_FOOD) -> pd.DataFrame:
    """Load pre-processed food-demand dataset."""
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  [load] food_demand.csv     -> {len(df):,} rows")
    return df


def load_water_requirement(filepath: str = RAW_WATER) -> pd.DataFrame:
    """Load water-requirement-per-crop dataset."""
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  [load] water_requirement   -> {len(df):,} rows")
    return df


# ===================================================================
#  2. Cleaning & filtering
# ===================================================================

def clean_crop_data(df: pd.DataFrame, year: int = 2019) -> pd.DataFrame:
    """
    Filter crop data for the specified year and 'Yield' element.
    Return a tidy DataFrame with columns: [crop, yield_kg_per_ha].
    """
    # Standardise column names (FAOSTAT uses mixed-case)
    df.columns = df.columns.str.strip()

    # Filter Element == "Yield"
    df = df[df["Element"].str.strip() == "Yield"].copy()

    # Filter Year
    df["Year"] = df["Year"].astype(int)
    df = df[df["Year"] == year].copy()

    # Drop rows with missing yield values
    df = df.dropna(subset=["Value"])

    # Aggregate: take the global average yield per crop item
    # (many countries; we want a single representative yield per crop)
    crop_yield = (
        df.groupby("Item", as_index=False)["Value"]
          .mean()
          .rename(columns={"Item": "crop", "Value": "yield_kg_per_ha"})
    )

    # Clean crop name whitespace
    crop_yield["crop"] = crop_yield["crop"].str.strip()

    print(f"  [clean] Yield entries for {year}: {len(crop_yield)} crops")
    return crop_yield


def clean_land_data(df: pd.DataFrame, year: int = 2019) -> pd.DataFrame:
    """
    Extract total agricultural land area (in hectares) for the given year.
    Returns a single-row DataFrame with column 'land_available_ha'.
    """
    df.columns = df.columns.str.strip()
    df["Year"] = df["Year"].astype(int)
    df = df[df["Year"] == year].copy()

    # Identify agricultural-land rows (FAOSTAT Item names vary;
    # look for items containing "Cropland" or "Agricultural" or similar)
    agri_keywords = ["Cropland", "Agricultural", "Arable"]
    mask = df["Item"].str.contains("|".join(agri_keywords), case=False, na=False)
    land_df = df[mask].copy()

    if land_df.empty:
        # Fallback: use all land rows and sum
        total_land_1000ha = df["Value"].sum()
    else:
        total_land_1000ha = land_df["Value"].sum()

    # Value is in '1000 ha' → convert to ha
    total_land_ha = total_land_1000ha * 1000.0
    print(f"  [clean] Total agricultural land ({year}): {total_land_ha:,.0f} ha")
    return total_land_ha


def clean_food_demand(df: pd.DataFrame) -> float:
    """
    Aggregate total food demand (food_supply across all countries/groups).
    Returns total demand in the same unit as the original column.
    """
    df.columns = df.columns.str.strip()
    total_demand = df["food_supply"].sum()
    print(f"  [clean] Total food demand: {total_demand:,.2f}")
    return total_demand


def clean_water_requirement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean DataFrame with [crop, water_requirement_m3_per_ha].
    """
    df.columns = df.columns.str.strip()
    df["crop"] = df["crop"].str.strip()
    df = df[["crop", "water_requirement_m3_per_ha"]].copy()
    df = df.dropna(subset=["water_requirement_m3_per_ha"])
    print(f"  [clean] Water requirement entries: {len(df)}")
    return df


# ===================================================================
#  3. Standardise crop names for merging
# ===================================================================

def standardize_crop_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(name.lower().strip().split())


def standardize_names_in_column(df: pd.DataFrame, col: str = "crop") -> pd.DataFrame:
    """Apply name standardisation in-place and return the dataframe."""
    df = df.copy()
    df["crop_std"] = df[col].apply(standardize_crop_name)
    return df


# ===================================================================
#  4. Merge datasets → model_input.csv
# ===================================================================

def merge_model_input(crop_yield: pd.DataFrame,
                      water_req: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join crop yield with water requirement on standardised crop name.
    Add a synthetic degradation index (higher water → higher degradation).
    """
    cy = standardize_names_in_column(crop_yield, "crop")
    wr = standardize_names_in_column(water_req,  "crop")

    merged = pd.merge(cy, wr, on="crop_std", how="inner",
                      suffixes=("_yield", "_water"))

    # Use the crop name from the water-requirement file (matches user spec)
    merged["crop"] = merged["crop_water"]

    # Convert yield from kg/ha to tonnes/ha for readability
    merged["yield"] = merged["yield_kg_per_ha"] / 1000.0

    merged["water_requirement"] = merged["water_requirement_m3_per_ha"]

    # Synthetic degradation index: normalise water requirement to [0, 1]
    max_water = merged["water_requirement"].max()
    if max_water > 0:
        merged["degradation_index"] = merged["water_requirement"] / max_water
    else:
        merged["degradation_index"] = 0.0

    result = merged[["crop", "yield", "water_requirement", "degradation_index"]].copy()
    result = result.reset_index(drop=True)
    print(f"  [merge] Model input rows: {len(result)}")
    return result


# ===================================================================
#  5. Build system_constraints.csv
# ===================================================================

def build_system_constraints(land_available_ha: float,
                             water_available_m3: float,
                             food_demand: float) -> pd.DataFrame:
    """
    Create a single-row DataFrame for system-level constraints.
    """
    constraints = pd.DataFrame([{
        "land_available":  land_available_ha,
        "water_available": water_available_m3,
        "food_demand":     food_demand,
    }])
    return constraints


# ===================================================================
#  6. Main preprocessing pipeline
# ===================================================================

def run_preprocessing() -> tuple:
    """
    Execute the full preprocessing pipeline:
      1. Load raw CSVs
      2. Clean / filter
      3. Merge
      4. Write model_input.csv & system_constraints.csv

    Returns
    -------
    model_input : pd.DataFrame
    system_constraints : pd.DataFrame
    """
    print("\n========== PREPROCESSING ==========\n")

    # --- Load ---
    crop_df  = load_crop_data()
    land_df  = load_land_data()
    food_df  = load_food_demand()
    water_df = load_water_requirement()

    # --- Clean ---
    crop_yield   = clean_crop_data(crop_df, year=2022)
    land_avail   = clean_land_data(land_df, year=2022)
    food_demand  = clean_food_demand(food_df)
    water_req    = clean_water_requirement(water_df)

    # --- Merge ---
    model_input = merge_model_input(crop_yield, water_req)

    # Water budget: scale average water requirement to land available
    if len(model_input) > 0:
        avg_water = model_input["water_requirement"].mean()
    else:
        avg_water = 6000.0  # fallback default
    water_available = avg_water * land_avail * 0.6   # 60 % of theoretical max

    system_constraints = build_system_constraints(
        land_available_ha=land_avail,
        water_available_m3=water_available,
        food_demand=food_demand,
    )

    # --- Save ---
    model_input.to_csv(OUT_MODEL_INPUT, index=False)
    system_constraints.to_csv(OUT_SYSTEM_CONSTRAINTS, index=False)

    print(f"\n  [OK] Saved {OUT_MODEL_INPUT}")
    print(f"  [OK] Saved {OUT_SYSTEM_CONSTRAINTS}")
    print("\n====================================\n")

    return model_input, system_constraints


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()
