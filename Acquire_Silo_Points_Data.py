import pandas as pd
import requests
import time
import os

# =====================================
# FILE PATHS
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
points_file = os.path.join(DATA_DIR, 'Unique_SILO_Points.xlsx')
output_folder = os.path.join(DATA_DIR, 'silo_csv')

# SILO asks for an email address as the username. Replace this with your own.
SILO_USERNAME = "your_email@example.com"

os.makedirs(output_folder, exist_ok=True)

# =====================================
# LOAD UNIQUE SILO POINTS
# =====================================
df = pd.read_excel(points_file)

required_cols = ["centre_longitude", "centre_latitude"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[required_cols].drop_duplicates().reset_index(drop=True)

print(f"Loaded {len(df)} unique SILO points")

# =====================================
# BASE URL + FIXED PARAMETERS
# =====================================
base_url = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"

fixed_params = {
    "format": "csv",
    "start": "20100101",
    "finish": "20260420",
    "username": SILO_USERNAME,
    "dataset": "Official",
    "comment": "r"
}

# =====================================
# DOWNLOAD LOOP
# =====================================
failed_points = []

for i, row in df.iterrows():
    lat = round(float(row["centre_latitude"]), 2)
    lon = round(float(row["centre_longitude"]), 2)

    params = fixed_params.copy()
    params["lat"] = f"{lat:.2f}"
    params["lon"] = f"{lon:.2f}"

    print(f"[{i+1}/{len(df)}] Downloading lat={lat:.2f}, lon={lon:.2f}")

    try:
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()

        text_preview = response.text[:500].lower()

        # Catch HTML / error pages instead of real CSV
        if "<html" in text_preview or "<!doctype" in text_preview or "error" in text_preview:
            print(f"  Failed: bad response for {lat:.2f}, {lon:.2f}")
            failed_points.append((lat, lon, "Bad response / HTML returned"))
            continue

        filename = f"SILO_{lat:.2f}_{lon:.2f}.csv"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"  Saved: {filepath}")

    except Exception as e:
        print(f"  Failed: {lat:.2f}, {lon:.2f} -> {e}")
        failed_points.append((lat, lon, str(e)))

    # Pause so you do not hammer the server
    time.sleep(1)

# =====================================
# SAVE FAILED POINTS
# =====================================
if failed_points:
    failed_df = pd.DataFrame(failed_points, columns=["latitude", "longitude", "reason"])
    failed_file = os.path.join(output_folder, "Failed_SILO_Points.xlsx")
    failed_df.to_excel(failed_file, index=False)
    print(f"\nSome downloads failed. See: {failed_file}")
else:
    print("\nAll downloads completed successfully.")