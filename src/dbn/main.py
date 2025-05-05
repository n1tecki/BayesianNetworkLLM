from src.dbn.dbn_training import dbn_train, flatten_df, predict_sepsis
from src.dbn.graph_visualisation import network_visualisation
import pandas as pd


LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", #"gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
df = pd.read_parquet("data/binned_train_data.parquet")


flat_df = flatten_df(df, LAB_COLS)
model, inference = dbn_train(flat_df, LAB_COLS)


network_visualisation(
    model,
    html_file="src/dbn/sepsis_dbn.html",
    notebook=False,
    physics="barnes_hut"
)


hadm_id = 22641185
new_patient = df.loc[hadm_id].reset_index(drop=True)[LAB_COLS]
pred = predict_sepsis(new_patient)
print(pred)