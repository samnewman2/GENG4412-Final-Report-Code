import pandas as pd
import numpy as np
import re
import os

# Includes Uncertainty Calculations

# =============================
# FILE PATH
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")   # all inputs and intermediate files live here
file_path = os.path.join(DATA_DIR, 'All_Stations_Discharge.xlsx')

# =============================
# UNCERTAINTY SYMBOLS
# =============================
uncertainty_map = {
    '"': 0.10,
    '#': 0.15,
    '$': 0.20,
    '&': 0.30
}

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# =============================
# PARSE FUNCTION
# =============================
def parse_value(cell):
    if pd.isna(cell):
        return (np.nan, 0)

    cell = str(cell).strip()

    symbol = None
    for s in uncertainty_map:
        if s in cell:
            symbol = s
            break

    num = re.sub(r'[^\d\.]', '', cell)
    value = float(num) if num != '' else np.nan

    uncertainty = uncertainty_map.get(symbol, 0)

    return value, uncertainty


# =============================
# LOAD DATA
# =============================
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# Fill station column
df['Station'] = df['Station'].ffill()

# Clean Year column
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Remove bad rows
df = df[df['Year'].notna()]

# DO NOT remove max year, otherwise 2025-2026 cannot be formed
# df = df[df['Year'] != df['Year'].max()]

df = df.sort_values(['Station', 'Year']).reset_index(drop=True)

# =============================
# SPLIT VALUES + UNCERTAINTY
# =============================
value_df = df.copy()
unc_df = df.copy()

for m in months:
    parsed = df[m].apply(parse_value)
    value_df[m] = parsed.apply(lambda x: x[0])
    unc_df[m] = parsed.apply(lambda x: x[1])

# =============================
# HYDROLOGICAL YEARS
# =============================
results = []

for station in value_df['Station'].dropna().unique():

    v = value_df[value_df['Station'] == station].reset_index(drop=True)
    u = unc_df[unc_df['Station'] == station].reset_index(drop=True)

    for i in range(len(v) - 1):

        year = int(v.loc[i, 'Year'])
        next_year = int(v.loc[i + 1, 'Year'])

        # Only use consecutive calendar years
        if next_year != year + 1:
            continue

        hydro_year = f"01/04/{year}-31/03/{next_year}"

        vals_1 = v.loc[i, ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        unc_1 = u.loc[i, ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

        vals_2 = v.loc[i + 1, ['Jan', 'Feb', 'Mar']]
        unc_2 = u.loc[i + 1, ['Jan', 'Feb', 'Mar']]

        values = pd.concat([vals_1, vals_2])
        uncertainties = pd.concat([unc_1, unc_2])

        total = values.sum(skipna=True)

        # Linear propagation
        total_uncertainty = (values * uncertainties).sum(skipna=True)

        # Percentage uncertainty
        percent_uncertainty = (total_uncertainty / total * 100) if total != 0 else 0

        results.append({
            'Station': int(station) if pd.notna(station) else station,
            'Year_Label': hydro_year,
            'Total Flow (ML)': round(total, 2),
            'Total Uncertainty (±ML)': round(total_uncertainty, 2),
            'Total Uncertainty (%)': round(percent_uncertainty, 2)
        })

# =============================
# OUTPUT
# =============================
result_df = pd.DataFrame(results)

output_path = file_path.replace(".xlsx", "_Hydrological_Years.xlsx")
result_df.to_excel(output_path, index=False)

print("✅ Done!")
print(result_df.head())