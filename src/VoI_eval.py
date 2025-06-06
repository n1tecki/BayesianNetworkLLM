
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.eval_utils.evaluation_time import (
    compare_single_threshold,
    compare_var_thresholds,
    compare_two_models,
    plot_accuracy_bars,
    plot_outcome_heatmaps,
)
from src.eval_utils.metrics import metrics_vs_threshold_runs
from src.eval_utils.VoI_curve_plot import plot_random_stay_trajectories

# --------------------------------------------------------------------------
# parameters – adjust here if needed
# --------------------------------------------------------------------------
VOI_THR = 0.50      # VoI probability threshold
SOFA_THR = 2         # SOFA-total threshold
PRED_FILE = Path("data/voi/voi_timelines_0_calibrated.json")

# --------------------------------------------------------------------------
# load data
# --------------------------------------------------------------------------
sofa_df = (
    pd.read_parquet("data/preprocessing/sofa_df_classified.parquet")
      .sort_index(level="timestamp")
)
sofa_df.index.names = ["hadm_id", "timestamp"]

df_test = pd.read_parquet("data/dbn/df_test.parquet").sort_values(
    ["hadm_id", "timestamp"]
)

with PRED_FILE.open() as f:
    voi_pred_dict = json.load(f)

test_ids = df_test.index.unique()

# --------------------------------------------------------------------------
# build per-stay timelines
# --------------------------------------------------------------------------
runs = []                 # (y_voi, ts_voi, y_sofa, ts_sofa)
septic_runs = []
correct_septic_runs = []

ground_truth = []         # 0/1 per stay
voi_sepsis_preds = []     # 0/1 per stay
sofa_sepsis_preds = []

for hadm in test_ids:

    # ---------------------------- VoI timeline ----------------------------
    all_steps = sorted(
        (r for r in voi_pred_dict[str(hadm)] if r["step"] >= 0),
        key=lambda d: d["step"],
    )
    y_voi = [r["p_post"] for r in all_steps]

    voi_rows = df_test.loc[hadm]
    if isinstance(voi_rows, pd.Series):        # single row edge case
        voi_rows = voi_rows.to_frame().T
    ts_voi = pd.to_datetime(voi_rows["timestamp"]).tolist()

    L = min(len(y_voi), len(ts_voi))           # keep them in lock-step
    y_voi, ts_voi = y_voi[:L], ts_voi[:L]

    # ---------------------------- SOFA timeline ---------------------------
    sofa_rows = sofa_df.loc[hadm]
    if isinstance(sofa_rows, pd.Series):
        sofa_rows = sofa_rows.to_frame().T
    sofa_rows = sofa_rows.sort_index()

    y_sofa  = sofa_rows["sofa_total"].tolist()
    ts_sofa = sofa_rows.index.tolist()

    L2 = min(len(y_sofa), len(ts_sofa))
    y_sofa, ts_sofa = y_sofa[:L2], ts_sofa[:L2]

    # ---------------------------- aggregate -------------------------------
    run_tuple = (y_voi, ts_voi, y_sofa, ts_sofa)
    runs.append(run_tuple)

    # ---------------------------- ground truth ----------------------------
    sepsis_vals = df_test.loc[hadm, "sepsis"]
    label = int(sepsis_vals.astype(int).max()) if isinstance(
        sepsis_vals, pd.Series
    ) else int(sepsis_vals)
    ground_truth.append(label)

    # ---------------------------- stay-level preds ------------------------
    voi_pred  = int(any(p >= VOI_THR  for p in y_voi))
    sofa_pred = int(any(s >= SOFA_THR for s in y_sofa))

    voi_sepsis_preds.append(voi_pred)
    sofa_sepsis_preds.append(sofa_pred)

    # ---------------------------- collect subsets -------------------------
    if label == 1:                     # septic stays only
        septic_runs.append(run_tuple)
        if voi_pred == 1 and sofa_pred == 1:
            correct_septic_runs.append(run_tuple)

# --------------------------------------------------------------------------
# inspection: label distribution
# --------------------------------------------------------------------------
print("\nStay-level label distribution in test set")
print(
    pd.Series(ground_truth)
      .value_counts()
      .sort_index()
      .rename({0: "non-sepsis", 1: "sepsis"})
)

# --------------------------------------------------------------------------
# Plot curve comparisons for a random stay
# --------------------------------------------------------------------------
plot_random_stay_trajectories(sofa_df, df_test, voi_pred_dict, test_ids, hadm=None, VOI_THR=VOI_THR, SOFA_THR=SOFA_THR)
# 21151580
# 29679030
# 26738143

# --------------------------------------------------------------------------
# lead-time analysis  (septic stays where both models fire)
# --------------------------------------------------------------------------
earlier, later, equal, d_steps, d_hours = compare_single_threshold(
    correct_septic_runs,
    dbn_thr=VOI_THR,         # function name kept from DBN, parameter is generic
    sofa_thr=SOFA_THR,
    show_plots=True,
)

print("\nLead-time analysis (ground-truth sepsis stays)")
print(f"  • VoI earlier  : {earlier:5d}")
print(f"  • SOFA earlier : {later:5d}")
print(f"  • tie          : {equal:5d}")

print(f"Median lead (lab values): {pd.Series(d_steps).median():.1f}")
print(f"Median lead (hours):      {pd.Series(d_hours).median():.1f}")
print(f"Mean lead   (lab values): {pd.Series(d_steps).mean():.1f}")
print(f"Mean lead   (hours):      {pd.Series(d_hours).mean():.1f}")

# --------------------------------------------------------------------------
# threshold sweeps & heat-maps  (septic subset)
# --------------------------------------------------------------------------
plot_outcome_heatmaps(correct_septic_runs)

compare_var_thresholds(correct_septic_runs, y_threshold=SOFA_THR)

# --------------------------------------------------------------------------
# accuracy, CIs, McNemar
# --------------------------------------------------------------------------
metrics_df, mcnemar_res = compare_two_models(
    ground_truth,
    voi_sepsis_preds,
    sofa_sepsis_preds,
    label_a="VoI",
    label_b="SOFA",
)

print("\nAccuracy metrics")
print(
    metrics_df[
        ["accuracy", "acc_CI_low", "acc_CI_high",
         "sensitivity", "specificity", "MCC"]
    ].round(3)
)

print("\nMcNemar χ² = {:.4f},   p = {:.4f}".format(
    mcnemar_res.statistic, mcnemar_res.pvalue)
)

plot_accuracy_bars(metrics_df, title="VoI vs SOFA – sepsis detection")

# --------------------------------------------------------------------------
# per-model confusion matrices
# --------------------------------------------------------------------------
def print_cm(y_true, y_pred, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion matrix – {label}")
    print("             Pred 0   Pred 1")
    print(f"True 0 |   {tn:5d}     {fp:5d}")
    print(f"True 1 |   {fn:5d}     {tp:5d}")

print_cm(ground_truth, voi_sepsis_preds,  "VoI")
print_cm(ground_truth, sofa_sepsis_preds, "SOFA")

# --------------------------------------------------------------------------
# metric curves vs. threshold
# --------------------------------------------------------------------------
voi_thrs  = np.arange(0.50, 0.95, 0.01)
metrics_vs_threshold_runs(runs, ground_truth, voi_thrs,  model="DBN")   # “DBN” pathway works for any prob series

sofa_thrs = np.arange(2, 15, 1)
metrics_vs_threshold_runs(runs, ground_truth, sofa_thrs, model="SOFA")
