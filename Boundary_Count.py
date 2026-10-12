import pandas as pd
import geopandas as gpd
import math
import os
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
from collections import defaultdict

# CALCULATES ALL THE SILO GRID CELLS THAT INTERSECT EACH CATCHMENT FOR EACH STREAM GAUGE SITE
#
# *** FIXED VERSION ***
# This is the script that originally GENERATES the SILO point files that
# Rainfall_Data_Calculations_Method_2.py and Stream_Gauge_Catchments_Map.py
# both depend on - so if this one searches around the wrong catchment
# shape, everything downstream inherits that gap even after those other
# scripts are fixed.
#
# The original script used
#     catchment_row = subcatchments.loc[catchment_index]
# where catchment_index came from `gpd.sjoin(gauges, subcatchments,
# predicate="within")` - "whichever polygon contains this gauge point".
# Every 'GS Catchment' polygon in this shapefile is only the small LOCAL
# increment draining directly to the gauge (e.g. Murray River's is 2.4
# km^2, Swan @ Walyunga's is 1.1 km^2), not the true upstream catchment
# (6,660 km^2 and ~119,000 km^2 respectively). That means the old script's
# search bounding box was far too small for most catchments, so it never
# even looked for SILO cells over most of the true catchment area - not
# just a weighting error, a genuine missing-data problem.
#
# This version reconstructs the TRUE upstream catchment for each gauge by
# tracing the shapefile's drainage network (ufi -> downstream fields)
# upstream from the gauge and merging every polygon that actually drains
# to it, THEN searches for SILO cells across that true extent. Verified
# against the shapefile's own official 'catch_full' total-catchment-area
# attribute (matches to within ~0.1% for every catchment tested).

# -----------------------------------
# 1. Load subcatchments shapefile
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
shp_path = os.path.join(DATA_DIR, 'shapefile', 'Hydrographic_Catchments_Subcatchments_DWER_030.shp')

if not os.path.exists(shp_path):
    folder = os.path.dirname(shp_path)
    if os.path.exists(folder):
        raise FileNotFoundError(
            f"Could not find:\n  {shp_path}\n\nFiles actually in that folder:\n  " +
            "\n  ".join(os.listdir(folder))
        )
    else:
        raise FileNotFoundError(f"The FOLDER itself doesn't exist:\n  {folder}")

subcatchments = gpd.read_file(shp_path)

print("Subcatchments CRS:")
print(subcatchments.crs)
print("\nSubcatchment columns:")
print(subcatchments.columns)

# -----------------------------------
# 2. Load stream gauge Excel file
# -----------------------------------
xlsx_path = os.path.join(DATA_DIR, 'Stream Gauge Data.xlsx')

if not os.path.exists(xlsx_path):
    folder = os.path.dirname(xlsx_path)
    if os.path.exists(folder):
        raise FileNotFoundError(
            f"Could not find:\n  {xlsx_path}\n\n.xlsx files actually in that folder:\n  " +
            "\n  ".join(f for f in os.listdir(folder) if f.lower().endswith(".xlsx"))
        )
    else:
        raise FileNotFoundError(f"The FOLDER itself doesn't exist:\n  {folder}")

df = pd.read_excel(xlsx_path)

print("\nGauge data:")
print(df.head())

if "Station" not in df.columns:
    raise ValueError(
        "Gauge data needs a 'Station' column (the AWRC station number) - the true "
        "catchment is looked up by station number now, not by spatial join."
    )


# -----------------------------------
# 3. RECONSTRUCT TRUE UPSTREAM CATCHMENTS (THE FIX)
#    Traces ufi -> downstream links backwards from each gauge station to
#    collect every polygon that actually drains to it, then merges them -
#    replaces the old sjoin(predicate="within") + .loc[catchment_index].
# -----------------------------------
def reconstruct_true_catchment(gdf, station_id):
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
        return None, None
    ufi = row.iloc[0]["ufi"]
    catchment_name = row.iloc[0]["catch_name"] if "catch_name" in row.columns else ""
    upstream_ufis = get_upstream(ufi)
    sub = gdf[gdf["ufi"].isin(upstream_ufis)]
    return unary_union(sub.geometry.values), catchment_name


df["Station"] = pd.to_numeric(df["Station"], errors="coerce")

# -----------------------------------
# 4. Convert Excel table into points (kept for the final plot only -
#    the catchment lookup itself no longer depends on the point location)
# -----------------------------------
gauges = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:7844"
)
gauges = gauges.to_crs(subcatchments.crs)

# -----------------------------------
# 5. SILO grid helpers
#    SILO centres are at exact 0.05 spacing
#    Example centre: -32.15, 116.05
# -----------------------------------
grid_size = 0.05
half_cell = grid_size / 2  # 0.025


def snap_down_to_centre(x, step=0.05):
    """Snap down to nearest valid SILO centre."""
    return round(math.floor(x / step) * step, 6)


def snap_up_to_centre(x, step=0.05):
    """Snap up to nearest valid SILO centre."""
    return round(math.ceil(x / step) * step, 6)


def frange(start, stop, step):
    """Range for floats, inclusive of stop (with tolerance)."""
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


# -----------------------------------
# 6. Build SILO cells grouped by Site Name, using the TRUE catchment
# -----------------------------------
summary_rows = []
gridpoint_rows = []
true_catchment_geoms = {}  # kept for the final plot

for i, gauge_row in df.iterrows():
    site_name = gauge_row["Site Name"] if "Site Name" in df.columns else f"Site_{i}"
    station = gauge_row["Station"]

    if pd.isna(station):
        print(f"  Skipping {site_name}: no Station number given")
        continue

    catchment_geom, catchment_name = reconstruct_true_catchment(subcatchments, int(station))
    if catchment_geom is None:
        print(f"  Skipping {site_name} (station {int(station)}): not found as a GS Catchment")
        continue

    true_catchment_geoms[station] = catchment_geom

    minx, miny, maxx, maxy = catchment_geom.bounds

    # Expand bounds slightly so edge cells are not missed
    centre_minx = snap_down_to_centre(minx - half_cell, grid_size)
    centre_maxx = snap_up_to_centre(maxx + half_cell, grid_size)
    centre_miny = snap_down_to_centre(miny - half_cell, grid_size)
    centre_maxy = snap_up_to_centre(maxy + half_cell, grid_size)

    lon_centres = frange(centre_minx, centre_maxx, grid_size)
    lat_centres = frange(centre_miny, centre_maxy, grid_size)

    intersecting_cells = []

    for centre_lon in lon_centres:
        for centre_lat in lat_centres:
            cell = box(
                centre_lon - half_cell,
                centre_lat - half_cell,
                centre_lon + half_cell,
                centre_lat + half_cell
            )

            if catchment_geom.intersects(cell):
                intersecting_cells.append({
                    "centre_longitude": round(centre_lon, 4),
                    "centre_latitude": round(centre_lat, 4),
                    "cell_west": round(centre_lon - half_cell, 4),
                    "cell_south": round(centre_lat - half_cell, 4),
                    "cell_east": round(centre_lon + half_cell, 4),
                    "cell_north": round(centre_lat + half_cell, 4)
                })

    print(f"  {site_name} (station {int(station)}): {len(intersecting_cells)} SILO cells found "
          f"over the true catchment (bounds {minx:.2f},{miny:.2f} to {maxx:.2f},{maxy:.2f})")

    # Summary row for each site
    summary_rows.append({
        "Site Name": site_name,
        "Station": station,
        "catchment_name": catchment_name,
        "west_most": round(minx, 4),
        "south_most": round(miny, 4),
        "east_most": round(maxx, 4),
        "north_most": round(maxy, 4),
        "centre_west_grid": centre_minx,
        "centre_south_grid": centre_miny,
        "centre_east_grid": centre_maxx,
        "centre_north_grid": centre_maxy,
        "silo_cell_count": len(intersecting_cells)
    })

    # One row per SILO point for this site
    point_number = 1
    for cell_info in intersecting_cells:
        gridpoint_rows.append({
            "Site Name": site_name,
            "Station": station,
            "catchment_name": catchment_name,
            "point_id": f"{int(station)}_{point_number}",
            "cell_west": cell_info["cell_west"],
            "cell_south": cell_info["cell_south"],
            "cell_east": cell_info["cell_east"],
            "cell_north": cell_info["cell_north"],
            "centre_longitude": cell_info["centre_longitude"],
            "centre_latitude": cell_info["centre_latitude"]
        })
        point_number += 1

# -----------------------------------
# 7. Convert to DataFrames
# -----------------------------------
summary_df = pd.DataFrame(summary_rows)
gridpoints_df = pd.DataFrame(gridpoint_rows)

# Add `catchment_index` for backwards compatibility with any downstream
# script that still expects that column name - it now just numbers each
# unique station rather than pointing at a shapefile row.
station_to_index = {s: i for i, s in enumerate(summary_df["Station"].unique())}
summary_df["catchment_index"] = summary_df["Station"].map(station_to_index)
gridpoints_df["catchment_index"] = gridpoints_df["Station"].map(station_to_index)

# Remove duplicate summaries if multiple rows somehow repeat
summary_df = summary_df.drop_duplicates(subset=["Site Name", "Station"]).reset_index(drop=True)

# Sort so each Site Name stays grouped together
gridpoints_df = gridpoints_df.sort_values(
    by=["Site Name", "Station", "centre_latitude", "centre_longitude"]
).reset_index(drop=True)

print("\nSite summary:")
print(summary_df.head())

print("\nSILO grid points grouped by Site Name:")
print(gridpoints_df.head(20))

# -----------------------------------
# 8. Save outputs
# -----------------------------------
summary_output = os.path.join(DATA_DIR, 'Site_SILO_Summary.xlsx')
gridpoints_output = os.path.join(DATA_DIR, 'Site_SILO_Grid_Points_Labelled.xlsx')

summary_df.to_excel(summary_output, index=False)
gridpoints_df.to_excel(gridpoints_output, index=False)

print(f"\nSaved summary to: {summary_output}")
print(f"Saved labelled SILO points to: {gridpoints_output}")

# -----------------------------------
# 9. Create UNIQUE SILO points file
# -----------------------------------
unique_points_df = gridpoints_df[[
    "centre_longitude",
    "centre_latitude"
]].drop_duplicates().reset_index(drop=True)

unique_points_df = unique_points_df.sort_values(
    by=["centre_latitude", "centre_longitude"]
).reset_index(drop=True)

print("\nUnique SILO points:")
print(unique_points_df.head())

unique_output = os.path.join(DATA_DIR, 'Unique_SILO_Points.xlsx')
unique_points_df.to_excel(unique_output, index=False)

print(f"Saved unique SILO points to: {unique_output}")

# -----------------------------------
# 10. Build true-catchment GeoDataFrame for the plot
#     (replaces the old "matched_subcatchments" built from catchment_index)
# -----------------------------------
true_catchments_gdf = gpd.GeoDataFrame(
    {"Station": list(true_catchment_geoms.keys())},
    geometry=list(true_catchment_geoms.values()),
    crs=subcatchments.crs
)

# -----------------------------------
# 11. Create SILO grid polygons
# -----------------------------------
grid_geoms = [
    box(
        row["cell_west"],
        row["cell_south"],
        row["cell_east"],
        row["cell_north"]
    )
    for _, row in gridpoints_df.iterrows()
]

grid_gdf = gpd.GeoDataFrame(
    gridpoints_df.copy(),
    geometry=grid_geoms,
    crs=subcatchments.crs
)

# -----------------------------------
# 12. Plot everything
# -----------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# True catchments
true_catchments_gdf.plot(
    ax=ax,
    edgecolor="black",
    facecolor="none",
    linewidth=1
)

# SILO grid
grid_gdf.plot(
    ax=ax,
    edgecolor="red",
    facecolor="none",
    linewidth=0.5
)

# Gauges
gauges.plot(
    ax=ax,
    color="blue",
    markersize=20
)

# -----------------------------------
# 13. Zoom to relevant area
# -----------------------------------
minx, miny, maxx, maxy = true_catchments_gdf.total_bounds

xpad = (maxx - minx) * 0.05
ypad = (maxy - miny) * 0.05

ax.set_xlim(minx - xpad, maxx + xpad)
ax.set_ylim(miny - ypad, maxy + ypad)

plt.title("True Catchments (reconstructed from drainage network) with SILO Grid Overlay")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig(os.path.join(DATA_DIR, "SILO_grid_overlay.png"), dpi=150)