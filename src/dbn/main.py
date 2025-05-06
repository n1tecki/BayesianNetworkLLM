from src.dbn.dbn_training import dbn_train, flatten_df
from dbn.evaluation import predict_df_data, steps_to_threshold, compare_single_threshold, compare_var_thresholds, compare_two_models, plot_accuracy_bars
from src.dbn.graph_visualisation import network_visualisation
from src.dbn.utils import split_train_test
import matplotlib.pyplot as plt
import pandas as pd


# ---------- LOAD DATA ----------------------------------------
# Read in data
LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "cns_score",
    #"gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
df = pd.read_parquet("data/binned_train_data.parquet")
df.index.name = 'hadm_id'

# Reading as int8 to reduce memory usage
cat_cols = ['sepsis'] + LAB_COLS
for c in cat_cols:
    df[c] = pd.Categorical(df[c]).codes.astype('int8')

# Split into training and test data
df_train, df_test = split_train_test(df, 
                        test_size=0.15, 
                        random_state=42
                    )
train_ids = df_train.index.unique()
test_ids = df_test.index.unique()


# ---------- TRAIN MODEL ----------------------------------------
# Flatten data and train model
flat_train_df = flatten_df(df_train, LAB_COLS)
model, inference = dbn_train(flat_train_df, 
                        LAB_COLS, 
                        CORRELATION_THRESHOLD = 0.4, 
                        alpha=1e-6
                    )

# Visualise model
network_visualisation(
    model,
    html_file="src/dbn/sepsis_dbn.html",
    notebook=False,
    physics="barnes_hut"
)


# ---------- PREEDICT DATA ----------------------------------------
# Predictions on new data
predictions_dict = predict_df_data(df_test, 
                        inference, 
                        LAB_COLS, 
                        save_path="data/predictions.json"
                    )


# ---------- EVALUATE PREDICTIONS ---------------------------------
# Load SOFA classifications
sofa_df = pd.read_parquet("data/semisupervised_df_classified")
sofa_df.index.name = 'hadm_id'

# Compare DBN probabilities to SOFA scores
dbn_thr = .7
sofa_thr = 2
runs = []
dbn_sepsis_preds = []
sofa_sepsis_preds = []
ground_truth_sepsis = []
for hadm in test_ids:
    # Sepsis timeline
    y_dbn = list(predictions_dict[hadm].values())
    y_sofa = sofa_df.loc[hadm, 'sofa'].tolist()
    runs.append((y_dbn, y_sofa))

    # Sepsis classification
    dbn_sepsis_preds.append(1 if any(x >= dbn_thr for x in y_dbn) else 0)
    sofa_sepsis_preds.append(1 if any(x >= sofa_thr for x in y_dbn) else 0)
    ground_truth_sepsis.append(sofa_df.loc[hadm, 'sepsis'].iloc[0])

# Single-threshold comparison
earlier, later, equal, deltas = compare_single_threshold(runs, dbn_thr, sofa_thr)

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

plot_accuracy_bars(metrics, title="DBN vs SOFA – Sepsis Detection")