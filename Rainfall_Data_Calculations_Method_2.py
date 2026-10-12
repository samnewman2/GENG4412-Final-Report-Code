import pandas as pd
import geopandas as gpd
import os
import re
from collections import defaultdict
from shapely.geometry import box
from shapely.ops import unary_union

# CALCULATES CATCHMENT RAINFALL BY WEIGHTING THE HYDROLOGICAL YEAR RAINFALL TOTALS FROM EACH SILO GRID CELL BY THE PROPORTION OF THAT CELL'S AREA THAT INTERSECTS THE CATCHMENT
#
# *** FIXED VERSION ***
# The original script picked the catchment polygon via
# `subcatchments_m.loc[catchment_index]` - a single row of the raw
# shapefile. Every 'GS Catchment' row's geometry is only the small LOCAL
# increment draining directly to that gauge (attribute `catch_sub`), not
# the true upstream catchment - for Murray River that's 2.4 km^2 against a
# true catchment of ~6,660 km^2; for Swan @ Walyunga it's 1.1 km^2 against
# ~119,000 km^2. Using that polygon here means the rainfall (P) was being
# area-averaged over the wrong (tiny) patch of ground, not the catchment
# that Q (streamflow) actually integrates over - a real mismatch in
# ET = P - Q/A, not just an approximation error.
#
# This version reconstructs the TRUE upstream catchment for each gauge by
# tracing the shapefile's drainage network (ufi -> downstream fields)
# upstream from the gauge and merging every contributing polygon, then
# uses THAT shape for the SILO-cell overlap weighting. Verified against
# the shapefile's own official 'catch_full' total-catchment-area attribute
# (matches to within ~0.1% for every catchment tested).

# ============================================================
# FILE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
shp_path = os.path.join(DATA_DIR, 'shapefile', 'Hydrographic_Catchments_Subcatchments_DWER_030.shp')
gridpoints_file = os.path.join(DATA_DIR, 'Site_SILO_Grid_Points_Labelled.xlsx')
rainfall_folder = os.path.join(DATA_DIR, 'silo_csv')
output_folder = DATA_DIR   # results are written next to the other inputs

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================
grid_size = 0.05
half_cell = grid_size / 2

date_col = "YYYY-MM-DD"
rain_col = "daily_rain"

# ============================================================
# 1. LOAD CATCHMENTS
# ============================================================
subcatchments = gpd.read_file(shp_path)

print("Subcatchments CRS:")
print(subcatchments.crs)
print("\nSubcatchment columns:")
print(subcatchments.columns)


# ============================================================
# 1B. RECONSTRUCT TRUE UPSTREAM CATCHMENTS (THE FIX)
#     Traces ufi -> downstream links backwards from each gauge station to
#     collect every polygon that actually drains to it, then merges them.
# ============================================================
def reconstruct_true_catchments(gdf, station_ids):
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
    gs = gs[gs["site_name"] != "WALYUNGA (pump site)"]  # duplicate/alt gauge point
    gs["win_ref"] = gs["win_ref"].astype(str)

    true_geoms = {}
    for station in station_ids:
        row = gs[gs["win_ref"] == str(station)]
        if len(row) == 0:
            print(f"  WARNING: station {station} not found as a GS Catchment - skipping")
            continue
        ufi = row.iloc[0]["ufi"]
        upstream_ufis = get_upstream(ufi)
        sub = gdf[gdf["ufi"].isin(upstream_ufis)]
        true_geoms[str(station)] = unary_union(sub.geometry.values)
        print(f"  station {station}: reconstructed from {len(sub)} upstream polygon(s), "
              f"area = {gpd.GeoSeries([true_geoms[str(station)]], crs=gdf.crs).to_crs(gs.estimate_utm_crs()).area.iloc[0]/1e6:.1f} km^2")
    return true_geoms


# ============================================================
# 2. LOAD LABELLED SILO GRID POINTS
#    This file should already contain:
#    Site Name, Station, catchment_index, centre_longitude, centre_latitude
# ============================================================
grid_df = pd.read_excel(gridpoints_file)

required_grid_cols = [
    "Site Name", "Station", "catchment_index",
    "centre_longitude", "centre_latitude"
]

for col in required_grid_cols:
    if col not in grid_df.columns:
        raise ValueError(f"Missing required column in gridpoints file: {col}")

# Make sure Station is numeric for correct sorting
grid_df["Station"] = pd.to_numeric(grid_df["Station"], errors="coerce")

# Remove duplicate point rows just in case
grid_df = grid_df.drop_duplicates(
    subset=["Site Name", "Station", "catchment_index", "centre_longitude", "centre_latitude"]
).reset_index(drop=True)

print(f"\nLoaded {len(grid_df)} labelled SILO point rows")

# Build the true catchment for every station that appears in the gridpoints file
# (note: `catchment_index` is no longer used to select geometry - only `Station` is)
station_ids = grid_df["Station"].dropna().unique().astype(int).astype(str)
print(f"\nReconstructing true catchments for {len(station_ids)} stations from the drainage network...")
true_catchment_geoms = reconstruct_true_catchments(subcatchments, station_ids)

# ============================================================
# 3. BUILD SILO CELL POLYGONS FROM CENTRES
# ============================================================
grid_df["cell_west"] = grid_df["centre_longitude"] - half_cell
grid_df["cell_south"] = grid_df["centre_latitude"] - half_cell
grid_df["cell_east"] = grid_df["centre_longitude"] + half_cell
grid_df["cell_north"] = grid_df["centre_latitude"] + half_cell

grid_geoms = [
    box(w, s, e, n)
    for w, s, e, n in zip(
        grid_df["cell_west"],
        grid_df["cell_south"],
        grid_df["cell_east"],
        grid_df["cell_north"]
    )
]

grid_gdf = gpd.GeoDataFrame(
    grid_df.copy(),
    geometry=grid_geoms,
    crs="EPSG:7844"
)

# Match catchment CRS if needed
if grid_gdf.crs != subcatchments.crs:
    grid_gdf = grid_gdf.to_crs(subcatchments.crs)

# ============================================================
# 4. PROJECT TO A METRIC CRS FOR AREA CALCULATIONS
# ============================================================
metric_crs = subcatchments.estimate_utm_crs()
if metric_crs is None:
    raise ValueError("Could not estimate a projected CRS for area calculations.")

subcatchments_m = subcatchments.to_crs(metric_crs)
grid_gdf_m = grid_gdf.to_crs(metric_crs)

# Reproject the reconstructed true catchment polygons into the same metric CRS too
true_catchment_geoms_m = {
    station: gpd.GeoSeries([geom], crs=subcatchments.crs).to_crs(metric_crs).iloc[0]
    for station, geom in true_catchment_geoms.items()
}

print(f"\nUsing projected CRS for area calculations: {metric_crs}")

# ============================================================
# 5. INTERSECT EACH SITE'S SILO CELLS WITH ITS TRUE CATCHMENT
#    Then calculate overlap weights
#    (grouped by Site Name + Station only - catchment_index is no longer
#    used to look up geometry, since that was the source of the bug)
# ============================================================
weight_rows = []

for (site_name, station), group in grid_gdf_m.groupby(["Site Name", "Station"]):
    station_str = str(int(station)) if pd.notna(station) else None
    if station_str not in true_catchment_geoms_m:
        print(f"  Skipping {site_name} (station {station}) - no reconstructed catchment available")
        continue

    catchment_geom = true_catchment_geoms_m[station_str]
    catchment_name = site_name  # true catchment may span multiple original subcatchment records

    total_overlap_area_m2 = 0
    temp_rows = []

    for _, cell_row in group.iterrows():
        intersection = cell_row.geometry.intersection(catchment_geom)
        overlap_area_m2 = intersection.area

        if overlap_area_m2 > 0:
            temp_rows.append({
                "Site Name": site_name,
                "Station": station,
                "catchment_name": catchment_name,
                "centre_longitude": round(float(cell_row["centre_longitude"]), 2),
                "centre_latitude": round(float(cell_row["centre_latitude"]), 2),
                "cell_area_km2": float(cell_row.geometry.area) / 1_000_000,
                "overlap_area_km2": float(overlap_area_m2) / 1_000_000
            })
            total_overlap_area_m2 += overlap_area_m2

    # Convert overlap area into weights that sum to 1 for that site/catchment
    for row in temp_rows:
        row["weight"] = (
            (row["overlap_area_km2"] * 1_000_000) / total_overlap_area_m2
            if total_overlap_area_m2 > 0 else 0
        )
        weight_rows.append(row)

weights_df = pd.DataFrame(weight_rows)

if weights_df.empty:
    raise RuntimeError("No overlap weights were created. Check CRS and catchment/grid alignment, "
                        "and check that every Station in the gridpoints file matched a reconstructed catchment.")

weights_df["Station"] = pd.to_numeric(weights_df["Station"], errors="coerce")

# Sort by station number lowest to highest
weights_df = weights_df.sort_values(
    by=["Station", "centre_latitude", "centre_longitude"]
).reset_index(drop=True)

# Save weights
weights_output = os.path.join(output_folder, "Catchment_SILO_Weights.xlsx")
weights_df.to_excel(weights_output, index=False)

print(f"\nSaved overlap weights to:\n{weights_output}")

# ============================================================
# 6. READ ALL SILO RAINFALL CSV FILES
#    Extract SILO point coordinates from filename
#    Example filename: SILO_-31.95_116.00.csv
# ============================================================
all_rain_rows = []
all_seasonal_rain_rows = []

pattern = re.compile(r"SILO_(-?\d+\.\d+)_(-?\d+\.\d+)\.csv$", re.IGNORECASE)

for file in os.listdir(rainfall_folder):
    if not file.lower().endswith(".csv"):
        continue

    match = pattern.match(file)
    if not match:
        print(f"Skipping filename that does not match expected SILO pattern: {file}")
        continue

    lat_from_file = round(float(match.group(1)), 2)
    lon_from_file = round(float(match.group(2)), 2)

    file_path = os.path.join(rainfall_folder, file)
    df = pd.read_csv(file_path)

    print(f"\nProcessing rainfall file: {file}")

    if date_col not in df.columns or rain_col not in df.columns:
        print(f"  Skipped: required columns not found in {file}")
        continue

    # Convert date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Convert rainfall
    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0)

    # Hydrological year
    df["Hydro_Year"] = df[date_col].apply(
        lambda x: x.year if x.month >= 4 else x.year - 1
    )

    # Wet/dry season split - Mediterranean climate pattern for SW WA
    # (May-September = wet season, rest of the year = dry season).
    # Adjust these month numbers if a different split is more appropriate.
    WET_SEASON_MONTHS = [5, 6, 7, 8, 9]
    df["Season"] = df[date_col].dt.month.apply(
        lambda m: "Wet" if m in WET_SEASON_MONTHS else "Dry"
    )

    # Remove incomplete / unwanted years
    df = df[~df["Hydro_Year"].isin([2009, 2026])]

    # Label
    df["Year_Label"] = df["Hydro_Year"].apply(
        lambda y: f"01/04/{y}-31/03/{y+1}"
    )

    # Sum rainfall by point + hydrological year (unchanged - still needed
    # for the annual water balance calculation elsewhere in the pipeline)
    yearly = df.groupby(["Hydro_Year", "Year_Label"], as_index=False)[rain_col].sum()

    yearly = yearly.rename(columns={
        rain_col: "Hydrological Rain Total (mm)"
    })

    yearly["centre_latitude"] = lat_from_file
    yearly["centre_longitude"] = lon_from_file

    all_rain_rows.append(yearly)

    # NEW: same aggregation, but split by season instead of collapsing
    # straight to one annual number - this is what gets lost by the
    # annual-only version above
    seasonal = df.groupby(["Hydro_Year", "Year_Label", "Season"], as_index=False)[rain_col].sum()
    seasonal = seasonal.rename(columns={rain_col: "Seasonal Rain Total (mm)"})
    seasonal["centre_latitude"] = lat_from_file
    seasonal["centre_longitude"] = lon_from_file
    all_seasonal_rain_rows.append(seasonal)

if not all_rain_rows:
    raise RuntimeError("No rainfall CSV files were successfully processed.")

rain_points_raw_df = pd.concat(all_rain_rows, ignore_index=True)

# ============================================================
# 6B. ADD STATION / SITE NAME TO RAIN POINTS
#     using the labelled SILO point file
# ============================================================
point_station_lookup = (
    grid_df[["Site Name", "Station", "centre_longitude", "centre_latitude"]]
    .drop_duplicates()
    .copy()
)

rain_points_df = pd.merge(
    rain_points_raw_df,
    point_station_lookup,
    on=["centre_longitude", "centre_latitude"],
    how="left"
)

rain_points_df["Station"] = pd.to_numeric(rain_points_df["Station"], errors="coerce")

# Sort nicely by station, then point, then year
rain_points_df = rain_points_df.sort_values(
    by=["Station", "centre_latitude", "centre_longitude", "Hydro_Year"]
).reset_index(drop=True)

# Create display copy for Excel output only
rain_points_display_df = rain_points_df.copy()

# Convert for display so blanks can be inserted safely
rain_points_display_df["Station"] = rain_points_display_df["Station"].apply(
    lambda x: "" if pd.isna(x) else str(int(x))
)
rain_points_display_df["Site Name"] = rain_points_display_df["Site Name"].astype("object")

# Show station and site name only once per SILO point block
repeat_mask = rain_points_display_df.duplicated(
    subset=["Station", "Site Name", "centre_latitude", "centre_longitude"]
)

rain_points_display_df.loc[repeat_mask, "Station"] = ""
rain_points_display_df.loc[repeat_mask, "Site Name"] = ""

# Reorder columns
rain_points_display_df = rain_points_display_df[
    [
        "Site Name",
        "Station",
        "centre_longitude",
        "centre_latitude",
        "Hydro_Year",
        "Year_Label",
        "Hydrological Rain Total (mm)"
    ]
]

rain_points_output = os.path.join(output_folder, "Hydrological_Rainfall_Totals_Per_SILO_Point.xlsx")
rain_points_display_df.to_excel(rain_points_output, index=False)

print(f"\nSaved per-point hydrological rainfall totals to:\n{rain_points_output}")

# ============================================================
# 7. JOIN RAINFALL TO CATCHMENT WEIGHTS
# ============================================================
merged_df = pd.merge(
    weights_df,
    rain_points_raw_df,
    on=["centre_longitude", "centre_latitude"],
    how="left"
)

if merged_df["Hydrological Rain Total (mm)"].isna().all():
    raise RuntimeError("Rainfall data did not join to the SILO weights. Check coordinate rounding and filenames.")

# Weighted rainfall contribution from each SILO cell
merged_df["Weighted Rainfall Contribution (mm)"] = (
    merged_df["Hydrological Rain Total (mm)"] * merged_df["weight"]
)

merged_output = os.path.join(output_folder, "Catchment_SILO_Weights_With_Rainfall.xlsx")
merged_df.to_excel(merged_output, index=False)

print(f"\nSaved merged weight + rainfall table to:\n{merged_output}")

# ============================================================
# 8. CALCULATE AREA-WEIGHTED CATCHMENT RAINFALL
#
#    IMPORTANT: pandas' .sum() skips NaN by default. If any SILO cell is
#    missing rainfall data for a given year, its weighted contribution
#    would otherwise silently vanish from the total (understating the
#    catchment rainfall for that year) with no warning at all. This
#    section checks for that and renormalizes weights over only the cells
#    that actually have data for each site-year, and prints a warning so
#    you know it happened and for which site/year.
# ============================================================
missing_mask = merged_df["Hydrological Rain Total (mm)"].isna()
if missing_mask.any():
    missing_summary = (
        merged_df[missing_mask]
        .groupby(["Site Name", "Hydro_Year"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "missing_weight_fraction"})
    )
    print("\nWARNING: some SILO cells are missing rainfall data for some years. "
          "Without correction this would silently understate the catchment total "
          "for that year. Affected site-years (missing_weight_fraction = fraction "
          "of the catchment's area with no data that year):")
    print(missing_summary.to_string(index=False))

    # Renormalize: for each site-year, rescale weights so they sum to 1 over
    # just the cells that actually have data, rather than silently treating
    # missing cells as contributing zero. Done with groupby().transform()
    # (fully vectorized) rather than .apply() so it isn't sensitive to
    # pandas-version differences in whether group keys are kept.
    group_cols = ["Site Name", "Station", "Hydro_Year", "Year_Label"]
    merged_df["_weight_if_valid"] = merged_df["weight"].where(~missing_mask, 0.0)
    merged_df["_valid_weight_sum"] = merged_df.groupby(group_cols)["_weight_if_valid"].transform("sum")
    merged_df["Weighted Rainfall Contribution (mm)"] = (
        merged_df["Hydrological Rain Total (mm)"] * merged_df["weight"] / merged_df["_valid_weight_sum"]
    )
    merged_df = merged_df.drop(columns=["_weight_if_valid", "_valid_weight_sum"])
else:
    print("\nNo missing rainfall data found in any SILO cell/year - no renormalization needed.")

catchment_rainfall_df = (
    merged_df.groupby(
        ["Site Name", "Station", "catchment_name", "Hydro_Year", "Year_Label"],
        as_index=False
    )["Weighted Rainfall Contribution (mm)"]
    .sum()
)

catchment_rainfall_df = catchment_rainfall_df.rename(columns={
    "Weighted Rainfall Contribution (mm)": "Catchment Hydrological Rainfall (mm)"
})

catchment_rainfall_df["Station"] = pd.to_numeric(catchment_rainfall_df["Station"], errors="coerce")

catchment_rainfall_df = catchment_rainfall_df.sort_values(
    by=["Station", "Hydro_Year"]
).reset_index(drop=True)

catchment_rainfall_output = os.path.join(output_folder, "Catchment_Hydrological_Rainfall.xlsx")
catchment_rainfall_df.to_excel(catchment_rainfall_output, index=False)

print(f"\nSaved final catchment rainfall totals to:\n{catchment_rainfall_output}")

# ============================================================
# 9. SAME WEIGHTING LOGIC, APPLIED TO SEASONAL TOTALS INSTEAD OF ANNUAL
#    Reuses the exact same spatial (Thiessen-style) weights already
#    computed above - only the TIME grouping differs (Season added
#    alongside Hydro_Year). Produces wet-season and dry-season
#    catchment rainfall, for testing whether seasonal similarity is a
#    better basis for similarity-weighted RF training than the current
#    whole-year average.
# ============================================================
seasonal_rain_raw_df = pd.concat(all_seasonal_rain_rows, ignore_index=True)

seasonal_merged_df = pd.merge(
    weights_df,
    seasonal_rain_raw_df,
    on=["centre_longitude", "centre_latitude"],
    how="left"
)

seasonal_missing_mask = seasonal_merged_df["Seasonal Rain Total (mm)"].isna()
if seasonal_missing_mask.any():
    seasonal_group_cols = ["Site Name", "Station", "Hydro_Year", "Year_Label", "Season"]
    seasonal_merged_df["_weight_if_valid"] = seasonal_merged_df["weight"].where(~seasonal_missing_mask, 0.0)
    seasonal_merged_df["_valid_weight_sum"] = seasonal_merged_df.groupby(seasonal_group_cols)["_weight_if_valid"].transform("sum")
    seasonal_merged_df["Weighted Seasonal Rainfall Contribution (mm)"] = (
        seasonal_merged_df["Seasonal Rain Total (mm)"] * seasonal_merged_df["weight"] / seasonal_merged_df["_valid_weight_sum"]
    )
    seasonal_merged_df = seasonal_merged_df.drop(columns=["_weight_if_valid", "_valid_weight_sum"])
else:
    seasonal_merged_df["Weighted Seasonal Rainfall Contribution (mm)"] = (
        seasonal_merged_df["Seasonal Rain Total (mm)"] * seasonal_merged_df["weight"]
    )

catchment_seasonal_df = (
    seasonal_merged_df.groupby(
        ["Site Name", "Station", "catchment_name", "Hydro_Year", "Year_Label", "Season"],
        as_index=False
    )["Weighted Seasonal Rainfall Contribution (mm)"]
    .sum()
    .rename(columns={"Weighted Seasonal Rainfall Contribution (mm)": "Catchment Seasonal Rainfall (mm)"})
)

# pivot so Wet and Dry become separate columns, one row per catchment-year -
# matching the shape of the existing annual file, ready to merge alongside it
catchment_seasonal_wide = catchment_seasonal_df.pivot_table(
    index=["Site Name", "Station", "catchment_name", "Hydro_Year", "Year_Label"],
    columns="Season", values="Catchment Seasonal Rainfall (mm)"
).reset_index()
catchment_seasonal_wide = catchment_seasonal_wide.rename(columns={
    "Wet": "Wet_Season_Rainfall_mm", "Dry": "Dry_Season_Rainfall_mm"
})
catchment_seasonal_wide["Station"] = pd.to_numeric(catchment_seasonal_wide["Station"], errors="coerce")
catchment_seasonal_wide = catchment_seasonal_wide.sort_values(by=["Station", "Hydro_Year"]).reset_index(drop=True)

catchment_seasonal_output = os.path.join(output_folder, "Catchment_Seasonal_Rainfall.xlsx")
catchment_seasonal_wide.to_excel(catchment_seasonal_output, index=False)
print(f"\nSaved catchment seasonal (wet/dry) rainfall totals to:\n{catchment_seasonal_output}")

print("\nDone.")