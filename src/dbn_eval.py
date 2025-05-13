from src.dbn_utils.evaluation import predict_df_data, steps_to_threshold, compare_single_threshold, compare_var_thresholds, compare_two_models, plot_accuracy_bars
import json
import pandas as pd



# Load SOFA and DBN classifications
sofa_df = pd.read_parquet("data/preprocessing/sofa_df_classified.parquet")
sofa_df.index.name = 'hadm_id'

df_test = pd.read_parquet("data/dbn/df_test.parquet")
test_ids = df_test.index.unique()

with open('data/predictions.json') as f:
    predictions_dict = json.load(f)


# Compare DBN probabilities to SOFA scores
# Neutral decision boundry would be 0.5
dbn_thr = .7
sofa_thr = 2
runs = []
correct_runs = []
dbn_sepsis_preds = []
sofa_sepsis_preds = []
correct_dbn_sepsis_preds = []
correct_sofa_sepsis_preds = []
ground_truth_sepsis = []

for hadm in test_ids:
    # Sepsis timeline
    y_dbn = list(predictions_dict[str(hadm)].values())
    y_sofa = sofa_df.loc[hadm, 'sofa_total'].tolist()
    runs.append((y_dbn, y_sofa))

    # Sepsis classification
    dbn_sepsis_preds.append(1 if any(x >= dbn_thr for x in y_dbn) else 0)
    sofa_sepsis_preds.append(1 if any(x >= sofa_thr for x in y_sofa) else 0)
    ground_truth_sepsis.append(sofa_df.loc[hadm, 'sepsis'].iloc[0])

# Get only correct predictions
for idx, truth in enumerate(ground_truth_sepsis):
    if (dbn_sepsis_preds[idx] == truth and
        sofa_sepsis_preds[idx] == truth):
        correct_dbn_sepsis_preds.append(runs[idx][0])
        correct_sofa_sepsis_preds.append(runs[idx][1])
        correct_runs.append(runs[idx])

# Single-threshold comparison
earlier, later, equal, deltas = compare_single_threshold(correct_runs, dbn_thr, sofa_thr)

# Sweep 0.5-1.0 and plot proportion where y_dbn is earlier
thr, earlier_p, later_p, equal_p = compare_var_thresholds(runs)

# Prediction accuracy
metrics, mcnemar_res = compare_two_models(ground_truth_sepsis,
                                          dbn_sepsis_preds, ground_truth_sepsis,
                                          label_a="DBN", label_b="SOFA")

print(metrics[["accuracy", "acc_CI_low", "acc_CI_high",
               "sensitivity", "specificity", "MCC"]].round(3))

print("\nMcNemar's χ² = {:.4f}, p-value = {:.4f}"
      .format(mcnemar_res.statistic, mcnemar_res.pvalue))

plot_accuracy_bars(metrics, title="DBN vs SOFA - Sepsis Detection")