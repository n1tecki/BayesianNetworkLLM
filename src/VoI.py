from src.VoI.voi_tools import value_of_information
import pickle, pandas as pd
import json

LAB_COLS = [
    "pf_ratio", "bilirubin_total", "creatinin",
    "cns_score", "mean_arterial_pressure", "platelet_count",
]

with open("data/dbn/inference.pkl", "rb") as f:
    inference = pickle.load(f)

df_test = pd.read_parquet("data/dbn/df_test.parquet")

summary = value_of_information(
    df_test,
    inference,
    LAB_COLS
)

with open("data/VoI/voi_timelines_double_layer.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

#print("Average lead-time (+ ⇒ earlier):", summary["lead_time"].mean())
