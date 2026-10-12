import pandas as pd
import numpy as np
import geopandas as gpd
import os
from collections import defaultdict
from shapely.ops import unary_union

# CALCULATES EVAPORATION ESTIMATES (P-Q/A) USING THE SECOND METHOD AND COMBINES ALL DATA INTO A SINGLE EXCEL FILE
#
# *** FIXED VERSION ***
# The original script used `stations["GgStn Catchment Area (km2)"]` as the
# denominator A in Evaporation = P - Q/A. That official area matches the
# true (drainage-network-traced) catchment area for 14 of 18 stations, but
# is badly wrong for a few: Swan River @ Walyunga's official area
# (18,629 km^2) is only ~15.6% of its true catchment (119,142 km^2) -
# because Q (measured streamflow) physically reflects the whole true
# catchment no matter what area you divide by, using the too-small
# official area inflates Q/A and understates Evaporation for that site
# (also smaller ~1.5-2.5% effects for Hotham and Murray Rivers).
#
# This version reconstructs the TRUE catchment area for every station by
# tracing the shapefile's drainage network (ufi -> downstream fields)
# upstream from the gauge and merging every polygon that actually drains
# to it - the same method used elsewhere in this pipeline, verified
# against the shapefile's own official 'catch_full' total-catchment-area
# attribute (matches to within ~0.1%).

# =============================
# FOLDER PATH
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
folder_path = DATA_DIR

# Catchment shapefile (DWER Hydrographic Catchments), used to reconstruct the true catchment area.
# the script will list the folder's contents if the file isn't found.
shp_path = os.path.join(DATA_DIR, 'shapefile', 'Hydrographic_Catchments_Subcatchments_DWER_030.shp')

# =============================
# FILE PATHS
# =============================
station_file = os.path.join(folder_path, "Stream Gauge Data.xlsx")
discharge_file = os.path.join(folder_path, "All_Stations_Discharge_Hydrological_Years.xlsx")
rainfall_file = os.path.join(folder_path, "Catchment_Hydrological_Rainfall.xlsx")
seasonal_rainfall_file = os.path.join(folder_path, "Catchment_Seasonal_Rainfall.xlsx")
bom_hindcast_file = os.path.join(folder_path, "BOM_Hindcast_Data.xlsx")
era5land_file = os.path.join(folder_path, "ERA5Land_Variables.xlsx")


def check_exists(path, exts=None):
    if os.path.exists(path):
        return
    folder = os.path.dirname(path)
    if os.path.exists(folder):
        files = os.listdir(folder)
        if exts:
            files = [f for f in files if f.lower().endswith(exts)]
        raise FileNotFoundError(
            f"Could not find:\n  {path}\n\nFiles actually in that folder:\n  " + "\n  ".join(files)
        )
    raise FileNotFoundError(f"The FOLDER itself doesn't exist:\n  {folder}")


for p, exts in [(station_file, (".xlsx",)), (discharge_file, (".xlsx",)),
                (rainfall_file, (".xlsx",)), (seasonal_rainfall_file, (".xlsx",)), (shp_path, None)]:
    check_exists(p, exts)

# =============================
# LOAD DATA
# =============================
stations = pd.read_excel(station_file)
discharge = pd.read_excel(discharge_file)
rainfall = pd.read_excel(rainfall_file)
seasonal_rainfall = pd.read_excel(seasonal_rainfall_file)

# BOM hindcast data is a newer, still-growing addition (built from
# extract_bom_hindcast_to_excel.py) - optional, since you're still
# gradually building up your collection of downloaded hindcast files.
# This script still runs fine without it, just without those two columns.
bom_hindcast = None
if os.path.exists(bom_hindcast_file):
    bom_hindcast = pd.read_excel(bom_hindcast_file)
    print(f"Loaded BOM hindcast data from: {bom_hindcast_file}")
else:
    print(f"NOTE: BOM hindcast file not found at {bom_hindcast_file} - "
          f"BOM_Hindcast_Rainfall_mm/BOM_Hindcast_Temperature_C/"
          f"BOM_Wet_Season_Temperature_C/BOM_Dry_Season_Temperature_C will be blank. "
          f"Run extract_bom_hindcast_to_excel.py to generate it.")

# Real, historical (not forecast) temperature/wind/radiation from ERA5-Land
# via Google Earth Engine - also optional, since it's a newer addition.
# Merged in later, after "Site Name" becomes available (unlike the other
# optional files, this one is keyed on Site Name, not Station, since the
# extraction script builds catchments directly by name).
era5land = None
if os.path.exists(era5land_file):
    era5land = pd.read_excel(era5land_file)
    print(f"Loaded ERA5-Land variables from: {era5land_file}")
else:
    print(f"NOTE: ERA5-Land file not found at {era5land_file} - "
          f"Temperature_C/Wind_Speed_ms/Solar_Radiation_MJ_m2 will be blank. "
          f"Run extract_era5land_variables.py to generate it.")


# =============================
# RECONSTRUCT TRUE CATCHMENT AREA (THE FIX)
#    Traces ufi -> downstream links backwards from each gauge station to
#    collect every polygon that actually drains to it, then merges them,
#    and reports the merged polygon's true area - replacing the "official"
#    GgStn Catchment Area, which is wrong for a few stations (see above).
# =============================
def reconstruct_true_area_km2(gdf, station_id):
    ds_map = dict(zip(gdf["ufi"], gdf["downstream"]))
    upstream_map = defaultdict(list)
    for ufi, ds in ds_map.items():
        if pd.isna(ufi) or pd.isna(ds):
            continue
        upstream_map[ds].append(ufi)

    def get_upstream(start_ufi):
        seen = {start_ufi}
        frontier = [start_ufi]
        while frontier:
            nxt = []
            for u in frontier:
                for up in upstream_map.get(u, []):
                    if up not in seen:
                        seen.add(up)
                        nxt.append(up)
            frontier = nxt
        return seen

    gs = gdf[gdf["type"] == "GS Catchment"].copy()
    gs = gs[gs["site_name"] != "WALYUNGA (pump site)"]
    gs["win_ref"] = gs["win_ref"].astype(str)

    row = gs[gs["win_ref"] == str(station_id)]
    if len(row) == 0:
        return np.nan
    ufi = row.iloc[0]["ufi"]
    upstream_ufis = get_upstream(ufi)
    sub = gdf[gdf["ufi"].isin(upstream_ufis)]
    geom = unary_union(sub.geometry.values)
    return gpd.GeoSeries([geom], crs=gdf.crs).to_crs(gs.estimate_utm_crs()).area.iloc[0] / 1e6


subcatchments = gpd.read_file(shp_path)

# =============================
# CLEAN STATION COLUMN
# =============================
stations["Station"] = stations["Station"].astype(str).str.strip()
discharge["Station"] = discharge["Station"].astype(str).str.strip()
rainfall["Station"] = rainfall["Station"].astype(str).str.strip()
seasonal_rainfall["Station"] = seasonal_rainfall["Station"].astype(str).str.strip()
if bom_hindcast is not None:
    bom_hindcast["Station"] = bom_hindcast["Station"].astype(str).str.strip()

# Fix area commas (kept for reference/comparison, no longer used in the
# Evaporation calculation itself - see "Area_official_km2" below)
stations["GgStn Catchment Area (km2)"] = (
    stations["GgStn Catchment Area (km2)"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

print("Reconstructing true catchment area for each station from the drainage network...")
stations["Area_true_km2"] = stations["Station"].apply(
    lambda s: reconstruct_true_area_km2(subcatchments, s)
)

# Flag and report any meaningful discrepancy between the official area and
# the true reconstructed area, so it's visible rather than silently swapped.
stations["area_pct_diff"] = (
    100 * (stations["Area_true_km2"] - stations["GgStn Catchment Area (km2)"])
    / stations["GgStn Catchment Area (km2)"]
)
flagged = stations[stations["area_pct_diff"].abs() > 0.5]
if len(flagged):
    print("\nWARNING: official catchment area differs meaningfully from the true "
          "(drainage-network-reconstructed) area for these stations - using the "
          "TRUE area for the Evaporation calculation below:")
    print(flagged[["Site Name", "Station", "GgStn Catchment Area (km2)",
                    "Area_true_km2", "area_pct_diff"]].to_string(index=False))
else:
    print("No meaningful area discrepancies found - official and true areas agree "
          "for every station.")

# Any station not found in the shapefile (e.g. Yarragil Brook) has no true
# area - fall back to the official area for those so they aren't silently
# dropped, but flag it.
missing_true_area = stations["Area_true_km2"].isna()
if missing_true_area.any():
    print("\nNOTE: no shapefile match found for these stations - falling back to "
          "the official area for them (results for these stations should be "
          "treated with the same caution as before this fix):")
    print(stations.loc[missing_true_area, ["Site Name", "Station"]].to_string(index=False))
    stations.loc[missing_true_area, "Area_true_km2"] = stations.loc[
        missing_true_area, "GgStn Catchment Area (km2)"]

# Drop unwanted columns
stations = stations.drop(columns=["Dates", "OBJECTID"], errors="ignore")

# Rename - "Area" now comes from the true reconstructed catchment, not the
# official (sometimes wrong) GgStn value. The official value is kept as
# "Area_official_km2" further down for transparency/comparison.
stations = stations.rename(columns={
    "GgStn Catchment Area (km2)": "Area_official_km2",
    "Area_true_km2": "Area"
})

# =============================
# KEEP REQUIRED COLUMNS
# =============================
rainfall = rainfall[[
    "Site Name",
    "Station",
    "Year_Label",
    "Catchment Hydrological Rainfall (mm)",
    "catchment_name"
]]

discharge = discharge[[
    "Station",
    "Year_Label",
    "Total Flow (ML)",
    "Total Uncertainty (±ML)",
    "Total Uncertainty (%)"
]]

# =============================
# MERGE DATASETS
# =============================
combined = pd.merge(
    rainfall,
    discharge,
    on=["Station", "Year_Label"],
    how="left"
)

seasonal_cols = seasonal_rainfall[["Station", "Year_Label", "Wet_Season_Rainfall_mm", "Dry_Season_Rainfall_mm"]]
combined = pd.merge(combined, seasonal_cols, on=["Station", "Year_Label"], how="left")

if bom_hindcast is not None:
    bom_cols = bom_hindcast[["Station", "Year_Label", "BOM_Hindcast_Rainfall_mm", "BOM_Hindcast_Temperature_C",
                              "BOM_Wet_Season_Temperature_C", "BOM_Dry_Season_Temperature_C"]]
    combined = pd.merge(combined, bom_cols, on=["Station", "Year_Label"], how="left")
else:
    combined["BOM_Hindcast_Rainfall_mm"] = np.nan
    combined["BOM_Hindcast_Temperature_C"] = np.nan
    combined["BOM_Wet_Season_Temperature_C"] = np.nan
    combined["BOM_Dry_Season_Temperature_C"] = np.nan

merged = pd.merge(
    combined,
    stations,
    on="Station",
    how="left"
)

# =============================
# FIX DUPLICATE COLUMN NAMES
# =============================
if "Site Name_x" in merged.columns:
    merged = merged.rename(columns={"Site Name_x": "Site Name"})

if "Site Name_y" in merged.columns:
    merged = merged.drop(columns=["Site Name_y"])

if era5land is not None:
    era5_cols = era5land[["Site Name", "Year_Label", "Temperature_C", "Wind_Speed_ms", "Solar_Radiation_MJ_m2"]]
    merged = pd.merge(merged, era5_cols, on=["Site Name", "Year_Label"], how="left")
else:
    merged["Temperature_C"] = np.nan
    merged["Wind_Speed_ms"] = np.nan
    merged["Solar_Radiation_MJ_m2"] = np.nan

# =============================
# RENAME COLUMNS
# =============================
merged = merged.rename(columns={
    "Catchment Hydrological Rainfall (mm)": "Hydrological Rain Total",
    "Total Uncertainty (±ML)": "Total Uncertainty (ML)"
})

# =============================
# CALCULATE EVAPORATION (now using the TRUE catchment area)
# =============================
merged["Evaporation (mm)"] = (
    merged["Hydrological Rain Total"] -
    (merged["Total Flow (ML)"] / merged["Area"])
).round(2)

merged["Evaporation/Rainfall (%)"] = np.where(
    merged["Hydrological Rain Total"] != 0,
    merged["Evaporation (mm)"] / merged["Hydrological Rain Total"] * 100,
    np.nan
).round(2)

# =============================
# PROPAGATE FLOW UNCERTAINTY INTO AN ET UNCERTAINTY
#    NOT the same number as "Total Uncertainty (%)" - that figure is a
#    percentage OF FLOW (Q), but ET is a different quantity entirely
#    (Evaporation = Rainfall - Q/Area). What genuinely propagates is the
#    ABSOLUTE uncertainty: since d(Evaporation)/d(Q) = -1/Area, the
#    uncertainty in Q (in ML) converts to mm using the exact same
#    ML-per-km^2-equals-mm relationship already used for Q/A itself.
#    Re-expressed as a percentage OF EVAPORATION (not of flow), this
#    comes out much smaller than the flow uncertainty percentage -
#    confirmed directly: flow uncertainty of 10-20% in this data
#    corresponds to an ET uncertainty of roughly 0.3-2.5% of ET, since
#    the streamflow term is typically a small fraction of total ET.
#
#    HONEST LIMITATION: this only captures the flow measurement's
#    contribution to ET uncertainty. Rainfall's own measurement
#    uncertainty isn't quantified anywhere in the source data, so it
#    can't be included here - this is a PARTIAL uncertainty estimate,
#    not the complete picture.
# =============================
merged["ET_Uncertainty (mm)"] = (
    merged["Total Uncertainty (ML)"] / merged["Area"]
).round(6)

merged["ET_Uncertainty (%)"] = np.where(
    merged["Evaporation (mm)"] != 0,
    100 * merged["ET_Uncertainty (mm)"].abs() / merged["Evaporation (mm)"].abs(),
    np.nan
).round(6)

# =============================
# SORT BY STATION NUMBER
# =============================
merged["Station_num"] = pd.to_numeric(merged["Station"], errors="coerce")

merged = merged.sort_values(
    by=["Station_num", "Year_Label"]
).reset_index(drop=True)

# =============================
# REMOVE REPEATED METADATA
# =============================
metadata_cols = [
    "Site Name",
    "Area",
    "Area_official_km2",
    "Latitude",
    "Longitude",
    "x",
    "y"
]

merged[metadata_cols] = merged.groupby("Station")[metadata_cols].transform(
    lambda x: x.where(x.index == x.index.min(), "")
)

# Show Station only once
merged["Station"] = merged.groupby("Station")["Station"].transform(
    lambda x: x.where(x.index == x.index.min(), "")
)

# =============================
# FINAL COLUMN NAMES
# =============================
merged = merged.rename(columns={
    "Area": "Area (km^2)(A)",
    "Area_official_km2": "Area_official_GgStn (km^2)",
    "Hydrological Rain Total": "Hydrological Rain Total (mm)(P)",
    "Total Flow (ML)": "Total Flow (ML)(Q)",
    "Evaporation (mm)": "Evaporation (mm)(P-Q/A)"
})

# =============================
# DROP HELPER COLUMN
# =============================
merged = merged.drop(columns=["Station_num", "area_pct_diff"], errors="ignore")

# =============================
# REORDER COLUMNS
#    Area_official_GgStn (km^2) is kept alongside the corrected area for
#    transparency - you can see directly where the two differ.
# =============================
merged = merged[
    [
        "Site Name",
        "Station",
        "x",
        "y",
        "Latitude",
        "Longitude",
        "Year_Label",
        "Area (km^2)(A)",
        "Area_official_GgStn (km^2)",
        "Hydrological Rain Total (mm)(P)",
        "Total Flow (ML)(Q)",
        "Total Uncertainty (ML)",
        "Total Uncertainty (%)",
        "Evaporation (mm)(P-Q/A)",
        "Evaporation/Rainfall (%)",
        "ET_Uncertainty (mm)",
        "ET_Uncertainty (%)",
        "Wet_Season_Rainfall_mm",
        "Dry_Season_Rainfall_mm",
        "BOM_Hindcast_Rainfall_mm",
        "BOM_Hindcast_Temperature_C",
        "BOM_Wet_Season_Temperature_C",
        "BOM_Dry_Season_Temperature_C",
        "Temperature_C",
        "Wind_Speed_ms",
        "Solar_Radiation_MJ_m2",
    ]
]

# =============================
# SAVE OUTPUT
# =============================
output_file = os.path.join(folder_path, "Combined_Data_Second_Method.xlsx")
merged.to_excel(output_file, index=False)

print("\nFinal dataset created:", output_file)