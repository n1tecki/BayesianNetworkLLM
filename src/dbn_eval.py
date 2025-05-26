# dbn_eval.py  – run with:  python dbn_eval.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.dbn_utils.metrics import metrics_vs_threshold_runs
from src.dbn_utils.evaluation_time import (
    compare_single_threshold,
    compare_var_thresholds,
    compare_two_models,
    plot_accuracy_bars,
    plot_outcome_heatmaps
)

# ------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------
DBN_THR  = 0.50          # DBN probability threshold
SOFA_THR = 2             # SOFA-total threshold

# ------------------------------------------------------------------
# load data
# ------------------------------------------------------------------
sofa_df = pd.read_parquet("data/preprocessing/sofa_df_classified.parquet")
sofa_df.index.names = ["hadm_id", "timestamp"]
sofa_df = sofa_df.sort_index(level="timestamp")

df_test = pd.read_parquet("data/dbn/df_test.parquet")
df_test = df_test.sort_values(["hadm_id", "timestamp"])

with Path("data/dbn/predictions_0_calibrated.json").open() as f:
    predictions_dict = json.load(f)

test_ids = df_test.index.unique()

# ------------------------------------------------------------------
# build per-stay timelines
# ------------------------------------------------------------------
runs, septic_runs, correct_septic_runs = [], [], []
dbn_sepsis_preds, sofa_sepsis_preds, ground_truth = [], [], []

for hadm in test_ids:

    # ------------------------------------------------------------
    # 1. build DBN & SOFA time-series (same as before)
    # ------------------------------------------------------------
    y_dbn = [pred for _, pred in sorted(
                 predictions_dict[str(hadm)].items(), key=lambda kv: int(kv[0]))]

    dbn_rows = df_test.loc[hadm]
    if isinstance(dbn_rows, pd.Series):
        dbn_rows = dbn_rows.to_frame().T
    dbn_rows = dbn_rows.sort_values("timestamp")
    ts_dbn   = pd.to_datetime(dbn_rows["timestamp"]).tolist()

    L = min(len(y_dbn), len(ts_dbn))
    y_dbn, ts_dbn = y_dbn[:L], ts_dbn[:L]

    sofa_rows = sofa_df.loc[hadm]
    if isinstance(sofa_rows, pd.Series):
        sofa_rows = sofa_rows.to_frame().T
    sofa_rows = sofa_rows.sort_index()
    y_sofa  = sofa_rows["sofa_total"].tolist()
    ts_sofa = sofa_rows.index.tolist()

    L2 = min(len(y_sofa), len(ts_sofa))
    y_sofa, ts_sofa = y_sofa[:L2], ts_sofa[:L2]

    run_tuple = (y_dbn, ts_dbn, y_sofa, ts_sofa)
    runs.append(run_tuple)

    # ------------------------------------------------------------
    # 2. stay-level ground truth  (from df_test only!)
    # ------------------------------------------------------------
    sepsis_vals = df_test.loc[hadm, "sepsis"]
    label = int(sepsis_vals.astype(int).max()) if isinstance(sepsis_vals, pd.Series) else int(sepsis_vals)
    ground_truth.append(label)

    # ------------------------------------------------------------
    # 3. stay-level predictions
    # ------------------------------------------------------------
    dbn_pred  = int(any(p >= DBN_THR  for p in y_dbn))
    sofa_pred = int(any(s >= SOFA_THR for s in y_sofa))
    dbn_sepsis_preds.append(dbn_pred)
    sofa_sepsis_preds.append(sofa_pred)

    # ------------------------------------------------------------
    # 4. collect subsets
    # ------------------------------------------------------------
    if label == 1:                       # all septic stays
        septic_runs.append(run_tuple)

        if dbn_pred == 1 and sofa_pred == 1:     # …and both models correct
            correct_septic_runs.append(run_tuple)

# ------------------------------------------------------------------
# sanity-check distribution
# ------------------------------------------------------------------
print("\nStay-level label distribution in test set")

print(pd.Series(ground_truth).value_counts().sort_index()
      .rename({0: "non-sepsis", 1: "sepsis"}))

# ------------------------------------------------------------------
# lead-time analysis  (septic only)
# ------------------------------------------------------------------
earlier, later, equal, d_steps, d_hours = compare_single_threshold(
        correct_septic_runs, dbn_thr=DBN_THR, sofa_thr=SOFA_THR, show_plots=True)

print("\nLead-time analysis (ground-truth sepsis stays)")
print(f"  • DBN earlier  : {earlier:5d}")
print(f"  • SOFA earlier : {later:5d}")
print(f"  • tie          : {equal:5d}")

print(f"Median lead (lab values): {pd.Series(d_steps).median():.1f}")
print(f"Median lead (hours): {pd.Series(d_hours).median():.1f}")

# ------------------------------------------------------------------
# DBN threshold sweep  (all stays)
# ------------------------------------------------------------------
plot_outcome_heatmaps(correct_septic_runs)

compare_var_thresholds(correct_septic_runs, y_threshold=SOFA_THR)
#compare_var_thresholds(correct_septic_runs, y_threshold=3)
#compare_var_thresholds(correct_septic_runs, y_threshold=4)
#compare_var_thresholds(correct_septic_runs, y_threshold=5)
#compare_var_thresholds(correct_septic_runs, y_threshold=6)
#compare_var_thresholds(correct_septic_runs, y_threshold=7)
#compare_var_thresholds(correct_septic_runs, y_threshold=8)

# ------------------------------------------------------------------
# accuracy, CIs, McNemar
# ------------------------------------------------------------------
metrics_df, mcnemar_res = compare_two_models(
        ground_truth, dbn_sepsis_preds, sofa_sepsis_preds,
        label_a="DBN", label_b="SOFA")

print("\nAccuracy metrics")
print(metrics_df[["accuracy", "acc_CI_low", "acc_CI_high",
                  "sensitivity", "specificity", "MCC"]].round(3))

print("\nMcNemar χ² = {:.4f},   p = {:.4f}"
      .format(mcnemar_res.statistic, mcnemar_res.pvalue))

plot_accuracy_bars(metrics_df, title="DBN vs SOFA – sepsis detection")

# ------------------------------------------------------------------
# per-model confusion matrices
# ------------------------------------------------------------------
def print_cm(y_true, y_pred, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion matrix – {label}")
    print("             Pred 0   Pred 1")
    print(f"True 0 |   {tn:5d}     {fp:5d}")
    print(f"True 1 |   {fn:5d}     {tp:5d}")

print_cm(ground_truth, dbn_sepsis_preds,  "DBN")
print_cm(ground_truth, sofa_sepsis_preds, "SOFA")


# --- DBN probability sweep ---------------------------------------
dbn_thrs = np.arange(0.50, 0.95, 0.01)
metrics_vs_threshold_runs(runs, ground_truth, dbn_thrs, model="DBN")

# --- SOFA total sweep --------------------------------------------
sofa_thrs = np.arange(2, 15, 1)
metrics_vs_threshold_runs(runs, ground_truth, sofa_thrs, model="SOFA")