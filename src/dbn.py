from src.dbn_utils.dbn_training import dbn_train, flatten_df, dbn_predict
from src.dbn_utils.graph_visualisation import network_visualisation
from src.dbn_utils.utils import create_train_test_set
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
                        CORRELATION_THRESHOLD = 0.4, 
                        alpha=1e-6
                    )

# Visualise model
network_visualisation(
    model,
    html_file="data/dbn/sepsis_dbn.html",
    notebook=False,
    physics="barnes_hut"
)

with open('data/dbn/model.pkl', 'wb') as f_model:
    pickle.dump(model, f_model)

with open('data/dbn/inference.pkl', 'wb') as f_inf:
    pickle.dump(inference, f_inf)



# ---------- PREDICT DATA ----------------------------------------
# Predictions on new data
predictions_dict = dbn_predict(df_test, 
    inference, 
    LAB_COLS
)

with open("data/dbn/predictions.json", "w") as f:
    json.dump(predictions_dict, f, indent=2)
