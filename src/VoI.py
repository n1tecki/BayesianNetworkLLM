from src.VoI.voi_evaluation import run_experiment
import pickle
import pandas as pd



LAB_COLS = [
        "pf_ratio", "bilirubin_total", "creatinin",
        "cns_score", "mean_arterial_pressure", "platelet_count",
    ]

with open('data/dbn/inference.pkl', 'rb') as f_inf:
    inference = pickle.load(f_inf)

df_test = pd.read_parquet("data/dbn/df_test.parquet")
test_ids = df_test.index.unique()



summary = run_experiment(df_test, inference, LAB_COLS, conf_threshold=0.7)

print("average lead-time (positive --> earlier detection):",
      summary['lead_time'].mean())
