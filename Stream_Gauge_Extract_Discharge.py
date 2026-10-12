import pandas as pd
import re
import glob
import os

# Extracts Data from htm files

def parse_line(line):
    tokens = []
    i = 0
    n = len(line)

    while i < n:
        char = line[i]

        # HANDLE [ ... ] → NaN
        if char == '[':
            while i < n and line[i] != ']':
                i += 1
            tokens.append(None)
            i += 1
            continue

        # START NUMBER
        if char.isdigit() or char == '.':
            token = ""

            while i < n and (line[i].isdigit() or line[i] == '.'):
                token += line[i]
                i += 1

            # skip spaces AFTER number
            j = i
            while j < n and line[j] == ' ':
                j += 1

            # attach symbol if exists
            if j < n and line[j] in ['"', '&', '#', '$', '!']:
                token += line[j]
                i = j + 1
            else:
                i = j

            tokens.append(token)
            continue

        i += 1

    return tokens


# 👉 Folder path (all .htm files)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
folder_path = os.path.join(DATA_DIR, 'discharge_htm', '*.htm')   # one .htm file per station

all_data = []

for file in glob.glob(folder_path):
    print(f"Processing: {file}")

    # Get station name from filename
    station_name = os.path.basename(file).replace(".htm", "")

    # Load HTML
    with open(file, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract <pre>
    pre_text = re.search(r"<pre>(.*?)</pre>", html, re.DOTALL).group(1)
    lines = pre_text.split("\n")

    for line in lines:
        line = line.strip()

        if re.match(r"^\d{4}", line):
            tokens = parse_line(line)

            if len(tokens) >= 14:
                year = int(tokens[0])

                if year >= 2010:
                    months = tokens[1:14]
                    all_data.append([station_name, year] + months)


# Create DataFrame
columns = ["Station", "Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Mean Monthly (ML)"]

df = pd.DataFrame(all_data, columns=columns)

# Fill missing years per station
final_list = []

for station in df["Station"].unique():
    temp = df[df["Station"] == station]

    all_years = pd.DataFrame({
        "Year": range(2010, temp["Year"].max() + 1)
    })

    temp = pd.merge(all_years, temp, on="Year", how="left")
    temp["Station"] = station

    final_list.append(temp)
formatted_rows = []

for station_df in final_list:
    station_df = station_df.sort_values("Year").reset_index(drop=True)

    first = True
    for _, row in station_df.iterrows():
        if first:
            formatted_rows.append(row)
            first = False
        else:
            row["Station"] = ""  # blank after first row
            formatted_rows.append(row)

final_df = pd.DataFrame(formatted_rows)

# Sort is already handled per station

# Save to Excel
output_path = os.path.join(DATA_DIR, 'All_Stations_Discharge.xlsx')

final_df.to_excel(output_path, index=False)

print("\n✅ Done! All files processed into one Excel.")