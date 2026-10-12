# Reconciling satellite and water-balance evapotranspiration (ET) in south-west Western Australia

Code for the GENG4412 final report *Reconciling Satellite and Water-Balance Evapotranspiration Estimates for Gauged Catchments in South-West Western Australia* by Sam Newman (University of Western Australia, 2026). Supervisors: Sally Thompson and Konrad Miotlinski.

The code builds an annual water-balance ET (ET = P − Q/A) for 17 gauged catchments (HY2011–HY2024, 238 catchment-years), uses it to correct annual MODIS16 and PML satellite ET with a Random Forest, reconciles the two, and spreads the result across the 46 eight-day periods of each hydrological year.

## Scripts and run order

All scripts read and write the `data/` folder next to them. Run them in this order.

| Step | Script | Reads (in `data/`) | Writes (in `data/`) |
|---|---|---|---|
| 1 | `Stream_Gauge_Extract_Discharge.py` | `discharge_htm/*.htm` (monthly discharge, one file per station) | `All_Stations_Discharge.xlsx` |
| 2 | `Discharge_Total_Year.py` | `All_Stations_Discharge.xlsx` | `All_Stations_Discharge_Hydrological_Years.xlsx` (annual flow and flow uncertainty) |
| 3 | `Boundary_Count.py` | `Stream Gauge Data.xlsx`, `shapefile/` | `Site_SILO_Grid_Points_Labelled.xlsx`, `Unique_SILO_Points.xlsx` |
| 4 | `Acquire_Silo_Points_Data.py` | `Unique_SILO_Points.xlsx` | `silo_csv/*.csv` (daily SILO rainfall per grid cell) |
| 5 | `Rainfall_Data_Calculations_Method_2.py` | grid points, `silo_csv/`, `shapefile/` | `Catchment_Hydrological_Rainfall.xlsx`, `Catchment_Seasonal_Rainfall.xlsx` |
| 6 | `Stream_Gauge_Data_Set_Code.py` | steps 2 and 5 outputs, `Stream Gauge Data.xlsx`, `shapefile/` | `Combined_Data_Second_Method.xlsx` (annual water balance) |
| 7 | `Extract_pml_modis16_eight_day.py` | `shapefile/` (Google Earth Engine) | `eight_day_satellite_ET_MODIS16.csv`, `eight_day_satellite_ET_PML.csv` (run once per product) |
| 8 | `Raw_Data_from_comparison.py` | steps 6 and 7 outputs | `merged_raw_data_for_verification.csv` |
| 9 | `rf_similarity_weighting_loco_nudging_mint_pipeline.py` | steps 6, 7 and 8 outputs | reconciled annual and 8-day ET, figures (in `results/`) |

Steps 1–8 prepare the data. Step 9 is the modelling pipeline.

## The modelling pipeline (step 9)

1. **Random Forest bias correction.** A Random Forest (150 trees, max depth 4, min leaf 5) predicts the satellite % bias against the water balance from rainfall features. It is evaluated leave-one-catchment-out, with training catchments weighted by seasonal rainfall similarity (bandwidths 100 mm for MODIS16, 30 mm for PML).
2. **Reconciliation.** Inverse-variance combination of the RF-corrected ET and the water balance. The code calls this step "nudging".
3. **Spreading to 8-day periods.** The annual gap is spread across the 8-day periods, scaled by a per-catchment trust factor (shrinkage K = 10, MinT-style).

The reconciled values use the water balance as an input, so scoring them against it is circular. See the report.

## Data you need to supply

The data files are not included. Request or download them and place them in `data/`:

- **Stream gauge discharge (monthly)**: **[agency / website, download date]**. Save each station's report as `data/discharge_htm/<station>.htm`.
- **`Stream Gauge Data.xlsx`**: gauge metadata (`Station`, `Site Name`, `Latitude`, `Longitude`, `GgStn Catchment Area (km2)`).
- **Catchment shapefile**: DWER Hydrographic Catchments – Subcatchments (DWER-030). Place the files in `data/shapefile/`.
- **SILO rainfall**: downloaded by step 4 from the SILO DataDrill service. Set your email in `SILO_USERNAME` in `Acquire_Silo_Points_Data.py`.
- **MODIS16 and PML ET**: extracted in step 7 from Google Earth Engine (`MODIS/061/MOD16A2GF` and `projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a`). Set your own Earth Engine project ID in `EE_PROJECT_ID`.

Streamflow data may carry licence conditions and is not redistributed here.

Optional columns: `Stream_Gauge_Data_Set_Code.py` can also merge BOM hindcast and ERA5-Land variables (`BOM_Hindcast_Data.xlsx`, `ERA5Land_Variables.xlsx`). The extraction scripts for these are not included, and the pipeline does not need them.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python Stream_Gauge_Extract_Discharge.py     # then each script in the order above
```

Python **3.10.11** was used.

## Known data limitations

- Flow months marked `!` (not reviewed / quality not known) in the discharge records are given zero uncertainty in `Discharge_Total_Year.py`. Most are for Canning River.
- Nine catchment-years have one month with no flow value, which `Discharge_Total_Year.py` skips, so those years' flow is slightly understated.
- `Raw_Data_from_comparison.py` is a rebuild of the script that originally produced `merged_raw_data_for_verification.csv`.

## Reproducibility

- Random seed fixed (`random_state=42`).
- The report's results come from the tagged version `v1.0-submission`.

## Contact

Sam Newman, **23457606@student.uwa.edu.au**
