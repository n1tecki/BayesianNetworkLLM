# ------------------------------------------------------------
# 0.  Imports + config
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import DBNInference

PARQUET_FILE = Path("data/semisupervised_df_classified.parquet")   # <-- adjust
LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count"
]
N_BINS = 10                   # 10 value buckets
MISSING_BIN = N_BINS          # bin index 10 == “missing”
# ------------------------------------------------------------
# 1.  Load + discretise (10 bins + missing)
# ------------------------------------------------------------
df = pd.read_parquet(PARQUET_FILE)                     # multi-index (hadm_id, sepsis, timestamp)
df = df.reset_index().sort_values(["hadm_id", "timestamp"])

# discretiser helper ----------------------------------------------------------
def discretise_with_missing(col: str, n_bins: int = N_BINS) -> None:
    mask_val = df[col].notna()
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    df.loc[mask_val, col] = kb.fit_transform(df.loc[mask_val, [col]]).astype(int)
    df.loc[~mask_val, col] = MISSING_BIN               # missing → 10
    df[col] = df[col].astype(int)

for lab in LAB_COLS:
    discretise_with_missing(lab)
# binary sepsis already clean, just cast to int
df["sepsis"] = df["sepsis"].astype(int)

# ------------------------------------------------------------
# 2.  Rolling 2-slice windows   (slice 0 = t, slice 1 = t+1)
# ------------------------------------------------------------
def build_windows(group: pd.DataFrame) -> list[dict]:
    records = []
    for i in range(len(group) - 1):
        rec = {}
        for col in ["sepsis"] + LAB_COLS:
            rec[(col, 0)] = group.iloc[i][col]       # time t
            rec[(col, 1)] = group.iloc[i + 1][col]   # time t+1
        records.append(rec)
    return records

records = (
    df.groupby("hadm_id", sort=False)
      .apply(build_windows)
      .explode()
      .dropna()                      # stays with only 1 row drop out
      .tolist()
)
train_df = pd.DataFrame(records)
train_df.columns = pd.MultiIndex.from_tuples(train_df.columns)

# ------------------------------------------------------------
# 3.  Build + fit the Dynamic Bayesian Network
# ------------------------------------------------------------
dbn = DBN()

# template edges --------------------------------------------------------------
dbn.add_edges_from([(("sepsis", 0), ("sepsis", 1))])           # hidden-state chain
for lab in LAB_COLS:
    dbn.add_edges_from([(("sepsis", 1), (lab, 1)),             # state → lab
                        ((lab, 0), (lab, 1))])                 # value carry-over

# fit (maximum-likelihood, all data observed)
dbn.fit(train_df, estimator="MLE")

# initialise slice 0 CPDs so inference knows the priors
dbn.initialize_initial_state()

# ------------------------------------------------------------
# 4.  Forward inference for ONE stay  (probability trajectory)
# ------------------------------------------------------------
hadm_example = df["hadm_id"].iloc[0]                 # ..pick any stay ID you like
example = df.query("hadm_id == @hadm_example").reset_index(drop=True)

# prepare evidence dict incrementally
evidence = {}
dbn_inf  = DBNInference(dbn)
posterior_curve = []

for t in example.index:
    for lab in LAB_COLS:                     # add this hour’s evidence
        evidence[(lab, t)] = example.loc[t, lab]
    q = dbn_inf.forward_inference([("sepsis", t)], evidence)
    prob_sepsis = q[("sepsis", t)].values[1]   # index 1 == sepsis ‘1’
    posterior_curve.append(prob_sepsis)

print(f"\nP(Sepsis=1) trajectory for stay {hadm_example}:")
for t, p in enumerate(posterior_curve):
    print(f"  t={t:02d}  P={p:7.4f}")
