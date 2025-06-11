from src.dbn_utils.dbn_training import dbn_train, flatten_df, dbn_predict
from src.dbn_utils.graph_visualisation import network_visualisation, network_slice_visualisations
from src.dbn_utils.utils import create_train_test_set
from src.dbn_utils.cpds_plot import plot_all_cpds_heatmap
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle


# ---------- LOAD DATA ----------------------------------------
# Read in data
LAB_COLS = [
        "pf_ratio", "bilirubin_total", "creatinin",
        "cns_score", "mean_arterial_pressure", "platelet_count",
    ]
df = pd.read_parquet("data/preprocessing/binned_train_data.parquet")
df_train, df_test = create_train_test_set(df=df, 
                                          cols=LAB_COLS, 
                                          test_size=0.3,
                                          export_path="data/dbn/")
train_ids = df_train.index.unique()
test_ids = df_test.index.unique()



# ---------- TRAIN MODEL ----------------------------------------
# Flatten data and train model
flat_train_df = flatten_df(df_train, LAB_COLS)
model, inference = dbn_train(flat_train_df, 
                        LAB_COLS,
                        alpha=1e-6,
                        pruning_delta=100,
                        use_bootstrap=True,
                        bootstrap_runs=10, 
                        bootstrap_conf=0.8
                    )

# Visualise model
network_visualisation(
    model,
    html_file="data/dbn/sepsis_dbn.html",
    notebook=False
)

network_slice_visualisations(
    model,
    base_html_file="data/dbn/dbn_slice"
)

with open('data/dbn/model.pkl', 'wb') as f_model:
    pickle.dump(model, f_model)

with open('data/dbn/inference.pkl', 'wb') as f_inf:
    pickle.dump(inference, f_inf)


# plot_all_cpds_heatmap(model.get_cpds(), indices=[0,3,7], n_cols=3)


# ---------- PREDICT DATA ----------------------------------------
# Predictions on new data
predictions_dict = dbn_predict(df_test, 
    inference, 
    LAB_COLS
)

with open("data/dbn/predictions_double_layer.json", "w") as f:
    json.dump(predictions_dict, f, indent=2)
