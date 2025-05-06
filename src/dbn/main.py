from src.dbn.dbn_training import dbn_train, flatten_df, predict_sepsis
from src.dbn.graph_visualisation import network_visualisation
from src.dbn.utils import split_train_test
import pandas as pd
from tqdm import tqdm
import json


LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", #"gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
df = pd.read_parquet("data/binned_train_data.parquet")
# Reading as int8 to reduce memory usage
cat_cols = ['sepsis'] + LAB_COLS
for c in cat_cols:
    df[c] = pd.Categorical(df[c]).codes.astype('int8')
df.index.name = 'hadm_id'
df_train, df_test = split_train_test(df, test_size=0.15, random_state=42)


flat_train_df = flatten_df(df_train, LAB_COLS)
model, inference = dbn_train(flat_train_df, LAB_COLS)


network_visualisation(
    model,
    html_file="src/dbn/sepsis_dbn.html",
    notebook=False,
    physics="barnes_hut"
)


predictions = {}
test_ids = df_test.index.unique()
for hadm_id in tqdm(test_ids):
    new_patient = df.loc[hadm_id].reset_index(drop=True)[LAB_COLS]
    pred = predict_sepsis(new_patient, inference, LAB_COLS)
    predictions[hadm_id] = pred

print(predictions)

with open("data/predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)