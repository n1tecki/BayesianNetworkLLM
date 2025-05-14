from src.VoI.voi_evaluation import run_experiment
import pickle, pandas as pd

LAB_COLS = [
    "pf_ratio", "bilirubin_total", "creatinin",
    "cns_score", "mean_arterial_pressure", "platelet_count",
]
MISSING_BIN = 3

with open("data/dbn/inference.pkl", "rb") as f:
    inference = pickle.load(f)

df_test = pd.read_parquet("data/dbn/df_test.parquet")

summary = run_experiment(
    df_test,
    inference,
    LAB_COLS,
    conf_threshold=0.7,
    missing_bin=MISSING_BIN
)

print("Average lead-time (+ ⇒ earlier):", summary["lead_time"].mean())
