"""
extract_pml_modis16_eight_day.py

Modified version of extract_pml_modis16_gee.py - instead of summing
each water year's 8-day ET composites into ONE annual total inside
Earth Engine (which is what the original script does, and is why
catchment_ET_PML_MODIS16.csv only ever had annual values), this keeps
each individual 8-day period as its own row - exactly what
the reconciliation pipeline needs as its EIGHT_DAY_DATA_PATH
input.

*** THIS IS SIGNIFICANTLY MORE EXPENSIVE THAN THE ANNUAL VERSION ***
A water year has ~46 8-day periods. Instead of one reduceRegions call
per catchment-year (what the original script does), this needs one
per catchment-PERIOD - roughly 46x more Earth Engine work per year.
Earth Engine applies compute quotas, so consider testing on a SHORT subset first (see WATER_YEARS and STATION_TO_NAME below -
both are trivially easy to temporarily shrink for a cheap test run
before committing to the full 15-year x 17-catchment extraction).

Uses Earth Engine's server-side .map() to reduce every image in the
collection in one batched request per water year, rather than a slow
client-side Python loop calling reduceRegions once per image - the
efficient, idiomatic way to do this in Earth Engine.

OUTPUT
------
  eight_day_satellite_ET.csv - one row per catchment per 8-day period,
  matching the reconciliation pipeline's expected input
  (Site Name, Hydro_Year, Period_Start_Date, Sat_ET_mm) - run once for
  MODIS16 and once for PML (SAT_PRODUCT below), or run twice and
  concatenate if you want both in one file.
"""

import os
import datetime
import ee
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from shapely.ops import unary_union

# ============================================================
# PATHS - same convention as extract_pml_modis16_gee.py
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
SHP_DIR = os.path.join(DATA_DIR, 'shapefile')
SHP_FILENAME = "Hydrographic_Catchments_Subcatchments_DWER_030.shp"
OUTPUT_DIR = DATA_DIR

STATION_TO_NAME = {
    '612022': 'Brunswick River - Sandalwood', '613002': 'Harvey River - Dingo Rd',
    '613007': 'Bancell Brook - Waterous', '614003': 'Marrinup Brook - Brookdale Siding',
    '614006': 'Murray River - Baden Powell', '614036': 'North Dandalup River - North Rd',
    '614037': 'Big Brook - Oneil Rd', '614196': 'Williams River - Saddleback Rd Bridge',
    '614224': 'Hotham River - Marradong Rd Bridge', '616001': 'Wooroloo Brook - Karls Ranch',
    '616002': 'Darkin River - Pine Plantation',
    '616012': 'Helena Brook - Trew Rd Gs', '616019': 'Brockman River - Yalliawirra',
    '616041': 'Wungong Brook - Vardi Rd', '616065': 'Canning River - Glen Eagle',
    '616178': 'Jane Brook - National Park', '616216': 'Helena River - Poison Lease Gs',
}

EE_PROJECT_ID = "YOUR_EARTH_ENGINE_PROJECT_ID"  # <-- replace with your own Google Earth Engine project

WATER_YEARS = [(2010, 2011), (2011, 2012), (2012, 2013), (2013, 2014), (2014, 2015), (2015, 2016),
               (2016, 2017), (2017, 2018), (2018, 2019), (2019, 2020), (2020, 2021), (2021, 2022),
               (2022, 2023), (2023, 2024), (2024, 2025)]  # <-- EDIT to temporarily shrink for a cheap test run if you want
# TESTING TIP: temporarily shrink this to e.g. WATER_YEARS[:1] and
# STATION_TO_NAME to just 1-2 entries for a cheap first test, given the
# cost warning above - confirm the output looks right before running
# the full extraction.

# Which product to extract this run - run once per product (each needs
# a different band-scaling approach, following the same logic already
# verified in extract_pml_modis16_gee.py)
SAT_PRODUCT = "MODIS16"  # "MODIS16" or "PML" - run once for each


# ============================================================
# 1. RECONSTRUCT TRUE CATCHMENTS (unchanged from the original script)
# ============================================================
def reconstruct_true_catchments(shp_path, station_to_name):
    gdf = gpd.read_file(shp_path)
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

    records, geoms = [], []
    for win_ref, name in station_to_name.items():
        row = gs[gs["win_ref"] == win_ref]
        if len(row) == 0:
            print(f"  {name} ({win_ref}): not found in shapefile - skipping")
            continue
        ufi = row.iloc[0]["ufi"]
        upstream_ufis = get_upstream(ufi)
        sub = gdf[gdf["ufi"].isin(upstream_ufis)]
        geom = unary_union(sub.geometry.values)
        records.append({"win_ref": win_ref, "Site Name": name})
        geoms.append(geom)
    return gpd.GeoDataFrame(records, geometry=geoms, crs=gdf.crs)


def init_earth_engine():
    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT_ID)


def catchments_to_ee_featurecollection(catch_gdf):
    features = []
    for _, row in catch_gdf.iterrows():
        geom_geojson = row.geometry.__geo_interface__
        ee_geom = ee.Geometry(geom_geojson)
        feat = ee.Feature(ee_geom, {"win_ref": row["win_ref"], "Site Name": row["Site Name"]})
        features.append(feat)
    return ee.FeatureCollection(features)


# ============================================================
# 2. PER-IMAGE (NOT annual-summed) SCALING - same scale factors already
#    verified in extract_pml_modis16_gee.py, just NOT summed across images
# ============================================================
def scale_modis16_image(img):
    et = img.select("ET").multiply(0.1)
    et = et.updateMask(img.select("ET").neq(32767))
    return et.rename("Sat_ET_mm").copyProperties(img, ["system:time_start"])


def scale_pml_image(img):
    # PML's own convention is a DAILY rate per 8-day period (not an
    # 8-day total like MODIS16) - multiply by 8 to get this period's
    # actual total, matching the official aggregation approach already
    # verified in extract_pml_modis16_gee.py, just applied per-image
    # instead of averaged-then-scaled-by-a-whole-year.
    total = img.select("Ec").add(img.select("Es")).add(img.select("Ei"))
    et = total.multiply(0.01).multiply(8)
    return et.rename("Sat_ET_mm").copyProperties(img, ["system:time_start"])


def get_water_year_collection(product, wy_start_year, wy_end_year):
    start = f"{wy_start_year}-04-01"
    end = f"{wy_end_year}-03-31"
    if product == "MODIS16":
        coll = ee.ImageCollection("MODIS/061/MOD16A2GF").filterDate(start, end).select("ET")
        return coll.map(scale_modis16_image)
    elif product == "PML":
        coll = ee.ImageCollection("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a").filterDate(start, end)
        return coll.map(scale_pml_image)
    else:
        raise ValueError(f"Unknown SAT_PRODUCT: {product}")


# ============================================================
# 3. EXTRACT EVERY 8-DAY PERIOD (not just the annual sum) - using
#    Earth Engine's server-side .map(), one batched request per water
#    year rather than one request per individual 8-day image
# ============================================================
def reduce_one_water_year(catch_fc, product, wy_start_year, wy_end_year):
    collection = get_water_year_collection(product, wy_start_year, wy_end_year)

    def reduce_one_image(img):
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        reduced = img.reduceRegions(collection=catch_fc, reducer=ee.Reducer.mean(), scale=500)

        def tag_date(feat):
            return feat.set("Period_Start_Date", date_str)

        return reduced.map(tag_date)

    # mapping reduceRegions across every image in ONE server-side call,
    # then flattening the resulting collection-of-collections into a
    # single flat FeatureCollection
    all_periods_fc = collection.map(reduce_one_image).flatten()
    return all_periods_fc.getInfo()


def extract_all(catch_fc, water_years, product):
    rows = []
    for wy_start, wy_end in water_years:
        print(f"Requesting {product}, water year {wy_start}-{wy_end} ")
        result_info = reduce_one_water_year(catch_fc, product, wy_start, wy_end)

        for i, feat in enumerate(result_info["features"]):
            props = feat["properties"]
            if i == 0:
                print(f"  Raw properties from Earth Engine (first period): {props}")

            sat_val = None
            for key, val in props.items():
                if key in ("win_ref", "Site Name", "Period_Start_Date"):
                    continue
                sat_val = val  # only one band present per product run
                break
            if sat_val is None:
                print(f"  WARNING: couldn't identify the ET value in {props} - check the raw "
                      f"properties above.")

            rows.append({
                "win_ref": props.get("win_ref"),
                "Site Name": props.get("Site Name"),
                "Hydro_Year": f"{wy_start}-{str(wy_end)[-2:]}",
                "Period_Start_Date": props.get("Period_Start_Date"),
                "Sat_ET_mm": sat_val,
            })
    return pd.DataFrame(rows)


def main():
    shp_path = os.path.join(SHP_DIR, SHP_FILENAME)

    print("Reconstructing true catchments...")
    catch_gdf = reconstruct_true_catchments(shp_path, STATION_TO_NAME)
    print(f"Reconstructed {len(catch_gdf)} / {len(STATION_TO_NAME)} catchments.\n")

    print("Initializing Earth Engine (a browser login window may open)...")
    init_earth_engine()

    catch_fc = catchments_to_ee_featurecollection(catch_gdf)

    print(f"Extracting {SAT_PRODUCT} at native 8-day resolution "
          f"(NOT aggregated to annual - see the cost warning in this script's docstring)...\n")
    out = extract_all(catch_fc, WATER_YEARS, SAT_PRODUCT)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"eight_day_satellite_ET_{SAT_PRODUCT}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(out)} rows)")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()