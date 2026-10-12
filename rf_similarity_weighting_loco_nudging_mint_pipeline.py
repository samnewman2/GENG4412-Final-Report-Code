"""
rf_loco_nudging_mint_pipeline.py

Extends rf_nudging_mint_pipeline.py to all 17 catchments, using Leave-
One-Catchment-Out (LOCO) instead of one fixed 70/30 split - and fixes
a real inconsistency that would exist otherwise: the earlier pipeline
used ONE fixed RF uncertainty (10.1% MODIS16, 7.3% PML) for every
catchment's nudging step, taken from a single split. But LOCO testing
proved RF's real accuracy varies hugely by catchment (4.5% to 45% for
MODIS16 alone) - using one number for everyone would ignore that.

STAGE 1 NOW USES SEASONAL SIMILARITY-WEIGHTED TRAINING (PRIMARY),
WITH WHOLE-YEAR SIMILARITY AS AN AUTOMATIC FALLBACK
--------------------------------------------------------------------------
Rather than judging similarity by whole-year average rainfall alone,
catchments are now compared using wet-season and dry-season rainfall
separately (2D distance instead of 1D) - found via a real bandwidth
sweep to outperform whole-year weighting for BOTH products once
properly tuned per-product (MODIS16: 13.1% -> 12.6%, PML: 8.7% -> 8.3%).
If the seasonal columns aren't present in Combined_Data_Second_
Method.xlsx yet, this automatically falls back to the original
whole-year similarity weighting (bandwidth=10) instead of failing.

THE THREE STAGES, NOW PER-CATCHMENT THROUGHOUT
--------------------------------------------------------------------------
1. RF PREDICTION (LOCO, similarity-weighted): for each of the 17
   catchments, train on the other 16 (weighted by climatic similarity),
   predict for the one held out - giving every catchment its own
   genuine, honest prediction AND its own measured accuracy.

2. NUDGING: blends that catchment's own RF prediction with water
   balance, using THAT CATCHMENT'S OWN measured RF accuracy (from
   stage 1) as the uncertainty weight - not a fixed number shared by
   all 17.

3. MinT SPREAD: same as before - the nudged annual target gets spread
   across the 46 8-day periods using each catchment's own trust factor.

OUTPUT
------
  <product>_loco_pipeline_results.csv
  <product>_loco_pipeline_summary.png
"""


# ==========================================================================
# IMPORTS
# ==========================================================================
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# SECTION 1 - FILE PATHS (relative to this script)
# ==========================================================================
# Paths are relative to this script. Put the input files in ./data/ (see README.md)
# or change the names below. Outputs are written to ./results/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ET_DATA_PATH = os.path.join(DATA_DIR, "merged_raw_data_for_verification.csv")
EIGHT_DAY_PATHS = {
    "MODIS16": os.path.join(DATA_DIR, "eight_day_satellite_ET_MODIS16.csv"),
    "PML": os.path.join(DATA_DIR, "eight_day_satellite_ET_PML.csv"),
}
ANNUAL_WB_PATH = os.path.join(DATA_DIR, "Combined_Data_Second_Method.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


# ==========================================================================
# SECTION 2 - MODEL AND UNCERTAINTY SETTINGS
# ==========================================================================
FEATURE_COLS = ["rainfall_anomaly_pct", "Rainfall_mm", "catchment_mean_rainfall"]
RF_PARAMS = dict(n_estimators=150, max_depth=4, min_samples_leaf=5, random_state=42)

RAINFALL_UNCERTAINTY_PCT = 3.0        # real, measured earlier in this project (SILO vs real BOM stations)
STORAGE_ASSUMPTION_UNCERTAINTY_PCT = 8.0   # stated assumption, not measured - chosen so the combined
                                            # total lands near the original flat 12% figure this replaces
MIN_EXPECTED_PERIODS = 40
SHRINKAGE_K = 10
WHOLE_YEAR_BANDWIDTH = {"MODIS16": 100, "PML": 500}  # [BANDWIDTH, mm] annual-rainfall fallback only, values from report Results 5.1

# Found via a real sweep against actual seasonal rainfall data - see
# rf_similarity_weighted_loco.py for the full sweep. Seasonal weighting
# beat whole-year weighting for BOTH products once properly tuned
# (MODIS16: 13.1% -> 12.6%, PML: 8.7% -> 8.3%), so this is now the
# primary method - whole-year weighting is kept only as an automatic
# fallback if the seasonal columns aren't present in the water balance
# source file yet.
SEASONAL_BANDWIDTH = {"MODIS16": 100, "PML": 30}  # [BANDWIDTH, mm] PRIMARY - values from report Results 5.1



# ==========================================================================
# SECTION 3 - DATA LOADING AND HELPER FUNCTIONS
# ==========================================================================
def compute_combined_wb_uncertainty(gauge_uncertainty_pct):
    """Replaces the old flat WATERBALANCE_UNCERTAINTY = 0.12 assumption
    with a genuinely catchment-year-specific value - see
    Combined_nudging_mint_reconciliation.py for the full derivation."""
    combined_pct = np.sqrt(
        gauge_uncertainty_pct ** 2 + RAINFALL_UNCERTAINTY_PCT ** 2 + STORAGE_ASSUMPTION_UNCERTAINTY_PCT ** 2
    )
    return combined_pct / 100


def load_et_uncertainty(path):
    """Loads the ET_Uncertainty (%) column added to Combined_Data_
    Second_Method.xlsx in Stream_Gauge_Data_Set_Code.py."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)
    df["Site Name"] = df["Site Name"].ffill()
    df["ET_Uncertainty (%)"] = pd.to_numeric(df["ET_Uncertainty (%)"], errors="coerce")
    return df[["Site Name", "Year_Label", "ET_Uncertainty (%)"]]



# ==========================================================================
# SECTION 4 - SIMILARITY WEIGHTS (used by Stage 1)
# ==========================================================================
def compute_whole_year_similarity_weight(train_df, held_out_mean, bandwidth):
    """Fallback only - used when seasonal rainfall data isn't available.
    Weights each training catchment by how similar its long-term annual
    rainfall is to the held-out catchment's."""
    distance = (train_df["catchment_mean_rainfall"] - held_out_mean).abs()
    weight = np.exp(-(distance ** 2) / (2 * bandwidth ** 2))
    return weight if weight.sum() > 1e-6 else None


def compute_seasonal_similarity_weight(train_df, held_out_wet, held_out_dry, bandwidth):
    """Primary method - distance is 2-dimensional (wet-season and
    dry-season rainfall separately), so two catchments with the same
    annual total but different seasonal timing are correctly treated
    as less similar, which the whole-year version can't distinguish."""
    wet_diff = train_df["catchment_mean_wet_rainfall"] - held_out_wet
    dry_diff = train_df["catchment_mean_dry_rainfall"] - held_out_dry
    distance = np.sqrt(wet_diff ** 2 + dry_diff ** 2)
    weight = np.exp(-(distance ** 2) / (2 * bandwidth ** 2))
    return weight if weight.sum() > 1e-6 else None


def load_seasonal_rainfall(path):
    """Loads Wet_Season_Rainfall_mm/Dry_Season_Rainfall_mm from
    Combined_Data_Second_Method.xlsx (added in Stream_Gauge_Data_Set_
    Code.py). Returns a dataframe with NaN seasonal columns if they
    don't exist in the source file at all (older version of the file,
    seasonal update not yet applied), so the caller's fallback check
    works whether the columns are missing entirely or just empty."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)
    df["Site Name"] = df["Site Name"].ffill()
    for col in ["Wet_Season_Rainfall_mm", "Dry_Season_Rainfall_mm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df[["Site Name", "Year_Label", "Wet_Season_Rainfall_mm", "Dry_Season_Rainfall_mm"]]


def extract_ending_year(value):
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


def load_annual_waterbalance_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)
    df["Site Name"] = df["Site Name"].ffill()
    df["Evaporation (mm)(P-Q/A)"] = pd.to_numeric(df["Evaporation (mm)(P-Q/A)"], errors="coerce")
    return df[["Site Name", "Year_Label", "Evaporation (mm)(P-Q/A)"]].rename(
        columns={"Evaporation (mm)(P-Q/A)": "waterbalance_annual_mm"})



# ==========================================================================
# SECTION 5 - STAGE 2 FUNCTION: inverse-variance reconciliation (report Equations 5-9)
# ==========================================================================
def nudge_toward_agreement(value_a, value_b, unc_a, unc_b):
    var_a = (unc_a * value_a) ** 2
    var_b = (unc_b * value_b) ** 2
    weight_a = 1 / var_a
    weight_b = 1 / var_b
    return (value_a * weight_a + value_b * weight_b) / (weight_a + weight_b)



# ==========================================================================
# SECTION 6 - STAGE 1: similarity-weighted Random Forest with leave-one-catchment-out (LOCO)
# ==========================================================================
def stage1_loco_rf(df, et_col, pct_diff_col, product_label):
    catchments = sorted(df["Site Name"].unique())
    catchment_means = df.groupby("Site Name")["catchment_mean_rainfall"].first()
    has_seasonal = "catchment_mean_wet_rainfall" in df.columns and df["catchment_mean_wet_rainfall"].notna().all()
    if has_seasonal:
        catchment_wet_means = df.groupby("Site Name")["catchment_mean_wet_rainfall"].first()
        catchment_dry_means = df.groupby("Site Name")["catchment_mean_dry_rainfall"].first()
    else:
        print(f"  NOTE: seasonal rainfall data not available - falling back to whole-year similarity weighting.")

    all_predictions = []
    for held_out in catchments:
        train_df = df[df["Site Name"] != held_out].copy()
        test_df = df[df["Site Name"] == held_out].copy()

        if has_seasonal:
            weight = compute_seasonal_similarity_weight(
                train_df, catchment_wet_means[held_out], catchment_dry_means[held_out],
                bandwidth=SEASONAL_BANDWIDTH[product_label])
        else:
            weight = compute_whole_year_similarity_weight(train_df, catchment_means[held_out],
                                                          bandwidth=WHOLE_YEAR_BANDWIDTH[product_label])

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(train_df[FEATURE_COLS], train_df[pct_diff_col], sample_weight=weight)
        predicted_bias = model.predict(test_df[FEATURE_COLS])
        test_df["rf_predicted_ET_mm"] = test_df[et_col] / (1 + predicted_bias / 100)
        test_df["raw_ET_mm"] = test_df[et_col]
        all_predictions.append(test_df)

    combined = pd.concat(all_predictions, ignore_index=True)

    per_catchment_mape = {}
    for site in catchments:
        sub = combined[combined["Site Name"] == site].dropna(subset=["rf_predicted_ET_mm", "WaterBalance_ET_mm"])
        mape = (100 * (sub["rf_predicted_ET_mm"] - sub["WaterBalance_ET_mm"]).abs() / sub["WaterBalance_ET_mm"]).mean()
        per_catchment_mape[site] = mape / 100

    return combined, per_catchment_mape



# ==========================================================================
# SECTION 7 - STAGE 2: reconcile RF estimate with water balance
# ==========================================================================
def stage2_nudge_per_catchment(df, per_catchment_mape):
    df = df.copy()
    df["rf_own_uncertainty"] = df["Site Name"].map(per_catchment_mape)
    df["wb_uncertainty"] = df["ET_Uncertainty (%)"].apply(compute_combined_wb_uncertainty)
    df["nudged_annual_mm"] = nudge_toward_agreement(
        df["rf_predicted_ET_mm"], df["WaterBalance_ET_mm"],
        df["rf_own_uncertainty"], df["wb_uncertainty"])
    return df



# ==========================================================================
# SECTION 8 - STAGE 3: MinT spreading to 8-day periods (trust factor with shrinkage)
# ==========================================================================
def estimate_shrunk_trust(annual_df, shrinkage_k=SHRINKAGE_K):
    annual_df = annual_df.copy()
    annual_df["incoherence_pct"] = 100 * (annual_df["waterbalance_annual_mm"] - annual_df["raw_annual_mm"]) / annual_df["waterbalance_annual_mm"]
    per_catchment = annual_df.groupby("Site Name")["incoherence_pct"].agg(["var", "count"]).reset_index()
    per_catchment.columns = ["Site Name", "own_variance", "n_years"]
    pooled_variance = annual_df["incoherence_pct"].var()
    per_catchment["alpha"] = per_catchment["n_years"] / (per_catchment["n_years"] + shrinkage_k)
    per_catchment["shrunk_variance"] = (per_catchment["alpha"] * per_catchment["own_variance"] +
                                         (1 - per_catchment["alpha"]) * pooled_variance)
    v = per_catchment["shrunk_variance"]
    v_min, v_max = v.min(), v.max()
    per_catchment["trust_factor"] = 1 - (v - v_min) / (v_max - v_min) if v_max > v_min else 1.0
    return per_catchment


def stage3_mint_spread(sat_8day_values, target_annual_value, trust_factor):
    sat_8day_values = np.asarray(sat_8day_values, dtype=float)
    m = len(sat_8day_values)
    gap = target_annual_value - sat_8day_values.sum()
    return sat_8day_values + (gap / m) * trust_factor



# ==========================================================================
# SECTION 9 - RUN ONE PRODUCT (runs Stages 1 -> 2 -> 3 and saves outputs)
# ==========================================================================
def run_for_product(product_label, et_col, pct_diff_col, annual_df, eight_day_path, wb_df, output_dir):
    print(f"\n{'=' * 60}\nProcessing {product_label}\n{'=' * 60}")

    stage1_df, per_catchment_mape = stage1_loco_rf(annual_df, et_col, pct_diff_col, product_label)
    print("  Per-catchment RF accuracy (used as each catchment's own nudging uncertainty):")
    for site, unc in sorted(per_catchment_mape.items(), key=lambda x: x[1]):
        print(f"    {site}: {unc:.1%}")

    stage2_df = stage2_nudge_per_catchment(stage1_df, per_catchment_mape)

    if not os.path.exists(eight_day_path) or wb_df is None:
        print(f"  No 8-day file (or water balance source) found - stopping after stage 2.")
        stage2_df.to_csv(os.path.join(output_dir, f"{product_label}_loco_annual_only.csv"), index=False)
        plot_summary(stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_pipeline_summary.png"))
        plot_connected_lines(stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_connected_lines.png"))
        plot_annual_per_catchment_over_time(stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_annual_over_time.png"))
        return stage2_df, None

    eight_day_df = pd.read_csv(eight_day_path)
    eight_day_df["_ending_year"] = eight_day_df["Hydro_Year"].apply(extract_ending_year)
    wb_df = wb_df.copy()
    wb_df["_ending_year"] = wb_df["Year_Label"].apply(extract_ending_year)

    full_annual_rows = []
    for (site, ending_year), group in eight_day_df.groupby(["Site Name", "_ending_year"]):
        if len(group) < MIN_EXPECTED_PERIODS:
            continue
        wb_match = wb_df[(wb_df["Site Name"] == site) & (wb_df["_ending_year"] == ending_year)]
        if wb_match.empty:
            continue
        full_annual_rows.append({"Site Name": site, "Year_Label": wb_match["Year_Label"].iloc[0],
                                  "raw_annual_mm": group["Sat_ET_mm"].sum(),
                                  "waterbalance_annual_mm": wb_match["waterbalance_annual_mm"].iloc[0]})
    full_annual_df = pd.DataFrame(full_annual_rows)
    trust_table = estimate_shrunk_trust(full_annual_df)
    trust_lookup = trust_table.set_index("Site Name")["trust_factor"].to_dict()

    detail_rows = []
    for _, row in stage2_df.iterrows():
        site, year_label = row["Site Name"], row["Year_Label"]
        ending_year = extract_ending_year(year_label)
        group = eight_day_df[(eight_day_df["Site Name"] == site) & (eight_day_df["_ending_year"] == ending_year)]
        if len(group) < MIN_EXPECTED_PERIODS:
            continue
        trust = trust_lookup.get(site, 0.5)

        reconciled = stage3_mint_spread(group["Sat_ET_mm"].values, row["nudged_annual_mm"], trust)
        for period_date, raw_val, recon_val in zip(group["Period_Start_Date"], group["Sat_ET_mm"], reconciled):
            detail_rows.append({
                "Site Name": site, "Year_Label": year_label, "Period_Start_Date": period_date,
                "raw_satellite_ET_mm": raw_val, "rf_predicted_annual_mm": row["rf_predicted_ET_mm"],
                "rf_own_uncertainty": row["rf_own_uncertainty"],
                "nudged_annual_target_mm": row["nudged_annual_mm"],
                "waterbalance_annual_mm": row["WaterBalance_ET_mm"],
                "trust_factor": trust, "final_reconciled_mm": recon_val,
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(output_dir, f"{product_label}_loco_pipeline_results.csv"), index=False)
    print(f"  Saved {product_label}_loco_pipeline_results.csv ({len(detail_df)} rows, "
          f"{detail_df['Site Name'].nunique()} catchments)")

    plot_summary(stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_pipeline_summary.png"))
    plot_connected_lines(stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_connected_lines.png"))
    plot_8day_all_catchments(detail_df, stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_8day_all_catchments.png"))
    plot_annual_per_catchment_over_time(stage2_df, product_label, os.path.join(output_dir, f"{product_label}_loco_annual_over_time.png"))
    return stage2_df, detail_df



# ==========================================================================
# SECTION 10 - FIGURES
# ==========================================================================
def plot_connected_lines(stage2_df, product_label, savepath):
    """Annual-level change: a thin line connecting each catchment-year's
    raw satellite position directly to its final nudged target, plotted
    against the water balance value - shows the actual size and
    direction of the annual correction for every individual point."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for _, row in stage2_df.iterrows():
        ax.plot([row["WaterBalance_ET_mm"], row["WaterBalance_ET_mm"]],
                [row["raw_ET_mm"], row["nudged_annual_mm"]],
                color="gray", lw=0.7, alpha=0.5, zorder=1)

    ax.scatter(stage2_df["WaterBalance_ET_mm"], stage2_df["raw_ET_mm"], color="#C0392B", alpha=0.7,
               label="Raw satellite (before)", s=45, zorder=2)
    ax.scatter(stage2_df["WaterBalance_ET_mm"], stage2_df["nudged_annual_mm"], color="#2E86AB", alpha=0.7,
               label="Final Reconciled target (after)", s=45, zorder=2)
    lims = [stage2_df["WaterBalance_ET_mm"].min() * 0.85, stage2_df["WaterBalance_ET_mm"].max() * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="1:1 (perfect)", zorder=1)

    ax.set_xlabel("Water balance ET (mm/yr)", fontsize=16)
    ax.set_ylabel("Annual estimate (mm/yr)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_title(f"{product_label}: annual change, every catchment-year\n"
                 f"raw satellite moved to its final reconciled target", fontsize=18)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)
    print(f"  Saved {os.path.basename(savepath)}")


def plot_8day_all_catchments(detail_df, stage2_df, product_label, savepath):
    """8-day level change: one panel per catchment (its largest-gap
    year), showing the raw 8-day series against the final reconciled
    series. Uses width-based halo styling so overlapping lines (e.g.
    low-trust catchments where raw and final nearly coincide) both
    stay visible rather than one hiding the other."""
    stage2_df = stage2_df.copy()
    stage2_df["gap"] = stage2_df["raw_ET_mm"] - stage2_df["WaterBalance_ET_mm"]
    stage2_df["abs_gap"] = stage2_df["gap"].abs()
    best_year_per_catchment = stage2_df.loc[stage2_df.groupby("Site Name")["abs_gap"].idxmax()].sort_values("Site Name")

    n = len(best_year_per_catchment)
    if n == 0:
        print("  No catchments to plot for the 8-day grid - skipping.")
        return
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(best_year_per_catchment.iterrows()):
        ax = axes[i]
        site, year = row["Site Name"], row["Year_Label"]
        sub = detail_df[(detail_df["Site Name"] == site) & (detail_df["Year_Label"] == year)].reset_index(drop=True)
        if len(sub) == 0:
            ax.axis("off")
            continue

        x = range(len(sub))
        ax.plot(x, sub["raw_satellite_ET_mm"], color="#C0392B", linewidth=4.5, alpha=0.5, label="Raw satellite", zorder=1)
        ax.plot(x, sub["final_reconciled_mm"], color="#2E86AB", linewidth=1.4, alpha=0.95,
                label="Final reconciled (RF+Reconciliation+MinT)", zorder=2)

        short_name = site.split(" - ")[0]
        trust = sub["trust_factor"].iloc[0]
        ax.set_title(f"{short_name}\n{year[6:10]}-{year[-2:]}, gap={row['gap']:.0f}mm, trust={trust:.2f}",
                     fontsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xticks([])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=18, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("8-day period through the hydrological year (1 to 46)", fontsize=18, y=0.065)
    fig.supylabel("8-day ET (mm)", fontsize=18)

    plt.suptitle(f"{product_label}: 8-day change, every catchment's largest-gap year\n"
                 "raw satellite vs the final reconciled result", fontsize=26)
    plt.tight_layout(rect=[0.02, 0.09, 1, 0.93])
    plt.savefig(savepath, dpi=130)
    plt.close(fig)
    print(f"  Saved {os.path.basename(savepath)}")


def plot_annual_per_catchment_over_time(stage2_df, product_label, savepath):
    """Year-by-year annual comparison, one panel per catchment: RF's
    Stage 1 prediction, the water balance value, and the final
    nudged estimate - confirmed mathematically that the final estimate
    always falls between RF's prediction and water balance (a genuine
    property of the inverse-variance weighted blend, verified across
    all 238 catchment-years with zero exceptions).

    The final estimate is drawn thicker and fully opaque so it stands
    out as the actual result of interest, while RF's prediction and
    water balance are drawn as thinner, more muted reference lines -
    the opposite emphasis from the 8-day overlap-handling plots, since
    here nothing needs a width-based halo (the three lines rarely sit
    exactly on top of each other at the annual level)."""
    stage2_df = stage2_df.copy()
    stage2_df["_end_year"] = stage2_df["Year_Label"].apply(extract_ending_year)
    stage2_df = stage2_df.sort_values(["Site Name", "_end_year"])

    catchments = sorted(stage2_df["Site Name"].unique())
    n = len(catchments)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, site in enumerate(catchments):
        ax = axes[i]
        sub = stage2_df[stage2_df["Site Name"] == site]
        years = sub["_end_year"].values
        x = range(len(years))

        ax.plot(x, sub["rf_predicted_ET_mm"], color="#C0392B", linewidth=3.2, alpha=0.35,
                label="RF prediction (Stage 1)", zorder=1)
        ax.plot(x, sub["WaterBalance_ET_mm"], color="#002CEE", linewidth=3.2, alpha=0.35,
                label="Water balance", zorder=1)
        ax.plot(x, sub["nudged_annual_mm"], color="#0E8131", linewidth=1.8, alpha=1.0,
                label="Final reconciled estimate", zorder=2)

        short_name = site.split(" - ")[0]
        ax.set_title(short_name, fontsize=17)
        ax.set_xticks(list(x)[::3])
        ax.set_xticklabels(years[::3], fontsize=13)
        ax.tick_params(axis="y", labelsize=13)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=18, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("Hydrological year (ending)", fontsize=18, y=0.065)
    fig.supylabel("Annual ET (mm)", fontsize=18)

    plt.suptitle(f"{product_label}: annual ET across all years, every catchment\n"
                 "RF prediction vs water balance vs the final reconciled estimate", fontsize=26)
    plt.tight_layout(rect=[0.02, 0.09, 1, 0.94])
    plt.savefig(savepath, dpi=130)
    plt.close(fig)
    print(f"  Saved {os.path.basename(savepath)}")


def plot_summary(stage2_df, product_label, savepath):
    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(stage2_df["WaterBalance_ET_mm"], stage2_df["nudged_annual_mm"],
                     c=stage2_df["rf_own_uncertainty"] * 100, cmap="RdYlGn_r", s=60, edgecolor="black", linewidth=0.5)
    lims = [stage2_df["WaterBalance_ET_mm"].min() * 0.85, stage2_df["WaterBalance_ET_mm"].max() * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("This catchment's own RF error (%), used as its RF uncertainty", fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    ax.set_xlabel("Water balance ET (mm/yr)", fontsize=16)
    ax.set_ylabel("Reconciled annual target (mm/yr)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_title(f"{product_label}: all 17 catchments, each reconciled using\nits own individually-measured RF accuracy", fontsize=18)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)
    print(f"  Saved {os.path.basename(savepath)}")



# ==========================================================================
# SECTION 11 - MAIN
# ==========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("RUN SETTINGS")
    print(f"  Seasonal bandwidth (primary): {SEASONAL_BANDWIDTH}")
    print(f"  Whole-year bandwidth (fallback only): {WHOLE_YEAR_BANDWIDTH}")
    print(f"  RF params: {RF_PARAMS}")
    print(f"  Rainfall uncertainty {RAINFALL_UNCERTAINTY_PCT}% | storage uncertainty {STORAGE_ASSUMPTION_UNCERTAINTY_PCT}%")
    df = pd.read_csv(ET_DATA_PATH)
    df["catchment_mean_rainfall"] = df.groupby("Site Name")["Rainfall_mm"].transform("mean")
    df["rainfall_anomaly_pct"] = 100 * (df["Rainfall_mm"] - df["catchment_mean_rainfall"]) / df["catchment_mean_rainfall"]

    wb_df = None
    if os.path.exists(ANNUAL_WB_PATH):
        wb_df = load_annual_waterbalance_xlsx(ANNUAL_WB_PATH)
        uncertainty_df = load_et_uncertainty(ANNUAL_WB_PATH)
        df = df.merge(uncertainty_df, on=["Site Name", "Year_Label"], how="left")
        n_missing = df["ET_Uncertainty (%)"].isna().sum()
        if n_missing:
            print(f"NOTE: {n_missing} rows had no matching ET uncertainty value.")

        seasonal_df = load_seasonal_rainfall(ANNUAL_WB_PATH)
        if seasonal_df[["Wet_Season_Rainfall_mm", "Dry_Season_Rainfall_mm"]].notna().any().any():
            df = df.merge(seasonal_df, on=["Site Name", "Year_Label"], how="left")
            df["catchment_mean_wet_rainfall"] = df.groupby("Site Name")["Wet_Season_Rainfall_mm"].transform("mean")
            df["catchment_mean_dry_rainfall"] = df.groupby("Site Name")["Dry_Season_Rainfall_mm"].transform("mean")
        else:
            print("NOTE: seasonal rainfall columns not found or empty - will fall back to whole-year similarity weighting.")

    for et_col, pct_diff_col, label in [
        ("MODIS16_ET_mm", "MODIS16_pct_diff", "MODIS16"),
        ("PML_ET_mm", "PML_pct_diff", "PML"),
    ]:
        eight_day_path = EIGHT_DAY_PATHS[label]
        run_for_product(label, et_col, pct_diff_col, df, eight_day_path, wb_df, OUTPUT_DIR)

    print("\nDONE.")


if __name__ == "__main__":
    main()