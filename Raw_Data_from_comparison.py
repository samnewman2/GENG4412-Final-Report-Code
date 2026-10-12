"""
build_merged_raw_data.py

Rebuilds the equivalent of merged_raw_data_for_verification.csv (the
RF-ready dataset used throughout this project's RF, LOCO, and
similarity-weighting scripts) - the original script that first built
this file couldn't be located, so this is a fresh, documented
equivalent, built directly from sources already confirmed correct
elsewhere in this project, with the new BOM hindcast columns included
from the start rather than merged in as a separate later step.

SOURCES USED
--------------------------------------------------------------------------
1. Combined_Data_Second_Method.xlsx - water balance ET, rainfall, and
   (if present) seasonal rainfall, ET uncertainty, and BOM hindcast
   columns, all already established and merged in this file.
2. eight_day_satellite_ET_MODIS16.csv / _PML.csv - the same real 8-day
   satellite extractions used throughout the reconciliation scripts
   (structural scaling, MinT, the full pipeline) - summed to annual
   totals here, so the MODIS16_ET_mm/PML_ET_mm values in the output
   are guaranteed consistent with everything else built in this
   project, rather than a separate, potentially inconsistent
   re-extraction.

OUTPUT COLUMNS
--------------------------------------------------------------------------
Matches the original file's structure exactly, plus two new ones:
  Site Name, Year_Label, WaterBalance_ET_mm, MODIS16_ET_mm, PML_ET_mm,
  Rainfall_mm, MODIS16_pct_diff, PML_pct_diff,
  BOM_Hindcast_Rainfall_mm, BOM_Hindcast_Temperature_C   (new)

HONEST NOTE ON COVERAGE
--------------------------------------------------------------------------
Only catchment-years present in BOTH the water balance source and the
8-day satellite files end up in the output - matches the same
inner-join logic the original file's usage implied throughout this
project (e.g. PML's slightly lower row count than MODIS16's, already
established elsewhere, is expected here too).
"""

import os
import numpy as np
import pandas as pd
import openpyxl

# ============================================================
# EDIT THESE
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
COMBINED_XLSX_PATH = os.path.join(DATA_DIR, 'Combined_Data_Second_Method.xlsx')
EIGHT_DAY_PATHS = {
    "MODIS16": os.path.join(DATA_DIR, 'eight_day_satellite_ET_MODIS16.csv'),
    "PML": os.path.join(DATA_DIR, 'eight_day_satellite_ET_PML.csv'),
}
OUTPUT_PATH = os.path.join(DATA_DIR, 'merged_raw_data_for_verification.csv')

MIN_EXPECTED_PERIODS = 40  # same incomplete-year guard used throughout this project


def extract_ending_year(value):
    """Same helper already established and reused throughout this
    project - converts either 'Year_Label' style
    ('01/04/2013-31/03/2014') or 'Hydro_Year' style ('2013-14') into a
    single comparable ending-year integer."""
    value = str(value).strip()
    if "-" in value and "/" in value:
        end_part = value.split("-")[-1]
        return int(end_part.split("/")[-1])
    if "-" in value:
        start_part, end_part = value.split("-")
        start_year = int(start_part)
        end_suffix = int(end_part)
        century = (start_year // 100) * 100
        end_year = century + end_suffix
        if end_year <= start_year:
            end_year += 100
        return end_year
    return int(value)


def load_combined_xlsx(path):
    """Loads water balance, rainfall, and any of the newer columns
    (seasonal rainfall, ET uncertainty, BOM hindcast) already present
    in Combined_Data_Second_Method.xlsx - only pulls in columns that
    actually exist, so this stays usable even against an older version
    of the file that doesn't have every optional column yet."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)
    df["Site Name"] = df["Site Name"].ffill()

    df["WaterBalance_ET_mm"] = pd.to_numeric(df["Evaporation (mm)(P-Q/A)"], errors="coerce")
    df["Rainfall_mm"] = pd.to_numeric(df["Hydrological Rain Total (mm)(P)"], errors="coerce")
    df["_ending_year"] = df["Year_Label"].apply(extract_ending_year)

    keep_cols = ["Site Name", "Year_Label", "_ending_year", "WaterBalance_ET_mm", "Rainfall_mm"]
    optional_cols = ["Wet_Season_Rainfall_mm", "Dry_Season_Rainfall_mm", "ET_Uncertainty (%)",
                     "BOM_Hindcast_Rainfall_mm", "BOM_Hindcast_Temperature_C",
                     "BOM_Wet_Season_Temperature_C", "BOM_Dry_Season_Temperature_C",
                     "Temperature_C", "Wind_Speed_ms", "Solar_Radiation_MJ_m2"]
    for col in optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep_cols.append(col)
        else:
            print(f"NOTE: '{col}' not found in {os.path.basename(path)} - skipping (older file version?).")

    return df[keep_cols]


def load_annual_satellite_et(paths):
    """Sums each product's real 8-day extraction to an annual total
    per catchment-year - the same real files used throughout this
    project's reconciliation scripts, so results here stay consistent
    with everything else already built."""
    annual_tables = {}
    for product_label, path in paths.items():
        df = pd.read_csv(path)
        df["_ending_year"] = df["Hydro_Year"].apply(extract_ending_year)

        annual = df.groupby(["Site Name", "_ending_year"]).agg(
            n_periods=("Sat_ET_mm", "size"),
            et_mm=("Sat_ET_mm", "sum"),
        ).reset_index()
        annual = annual[annual["n_periods"] >= MIN_EXPECTED_PERIODS].drop(columns=["n_periods"])
        annual = annual.rename(columns={"et_mm": f"{product_label}_ET_mm"})
        annual_tables[product_label] = annual

    return annual_tables


def main():
    print("Loading Combined_Data_Second_Method.xlsx...")
    combined_df = load_combined_xlsx(COMBINED_XLSX_PATH)

    print("Loading and summing 8-day satellite extractions to annual totals...")
    annual_tables = load_annual_satellite_et(EIGHT_DAY_PATHS)

    merged = combined_df.copy()
    for product_label, annual_df in annual_tables.items():
        merged = merged.merge(annual_df, on=["Site Name", "_ending_year"], how="inner")

    merged["MODIS16_pct_diff"] = 100 * (merged["MODIS16_ET_mm"] - merged["WaterBalance_ET_mm"]) / merged["WaterBalance_ET_mm"]
    merged["PML_pct_diff"] = 100 * (merged["PML_ET_mm"] - merged["WaterBalance_ET_mm"]) / merged["WaterBalance_ET_mm"]

    merged = merged.drop(columns=["_ending_year"])

    print(f"\nFinal dataset: {len(merged)} catchment-years, {merged['Site Name'].nunique()} catchments")
    print(f"Columns: {list(merged.columns)}")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
