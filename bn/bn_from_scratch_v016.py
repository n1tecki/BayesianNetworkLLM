import pandas as pd
from itertools import permutations
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import TabularCPD
import numpy as np


# — Configuration ——————————————————————————————————————————
# Flatten dataframe into two time slices (t0 -> t1)
LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]

df = pd.read_parquet("data/binned_train_data.parquet")
df = df.groupby(level="hadm_id").filter(lambda group: len(group) > 1)
def two_slice(df, labs):
    base = ["sepsis"] + labs
    d = df.reset_index().sort_values(["hadm_id", "timestamp"])
    now  = d[base].add_suffix("_1")
    prev = d.groupby("hadm_id")[base].shift(1).add_suffix("_0")
    return pd.concat([prev, now], axis=1).dropna().reset_index(drop=True)
flat_df = two_slice(df, LAB_COLS)




# — Select possibel edges —————————————————————————————————
CORRELATION_THRESHOLD = 0.4
corr_matrix = flat_df[[f"{col}_1" for col in ["sepsis"] + LAB_COLS]].corr(method="spearman").abs()
whitelist_edges = set()

# sepsis(1) -> every lab(1)
whitelist_edges |= {("sepsis_1", f"{lab}_1") for lab in LAB_COLS} 

# Edges between each lab t0 and each lab t1
for var in ["sepsis"] + LAB_COLS:
    whitelist_edges.add((f"{var}_0", f"{var}_1"))

# lab-lab edges at t=1 with ρ ≥ threshold
for l1 in LAB_COLS:
    for l2 in LAB_COLS:
        if l1 != l2 and corr_matrix.loc[f"{l1}_1", f"{l2}_1"] >= CORRELATION_THRESHOLD:
            whitelist_edges.add((f"{l1}_1", f"{l2}_1"))

# Define forbidden edges (blacklist)
all_vars = list(flat_df.columns)
all_possible_edges = set(permutations(all_vars, 2))
blacklist_edges = all_possible_edges - whitelist_edges


# — Structure Learning ———————————————————————————————————————
structure_estimator = HillClimbSearch(flat_df)
estimated_model = structure_estimator.estimate(
    scoring_method=BicScore(flat_df),
    white_list=list(whitelist_edges),
    black_list=list(blacklist_edges),
)

# keep only edges WITH child in slice 1  (child endswith '_1')
in_slice_edges = [(u, v) for u, v in estimated_model.edges() if v.endswith("_1")]

print(estimated_model.edges())




# — DBN Model Creation ———————————————————————————————————————
model = DBN()

# Adding persistence edges
edges = []
for u, v in estimated_model.edges():
    u_var, u_time = u, int(u.rsplit('_', 1)[1])
    v_var, v_time = v, int(v.rsplit('_', 1)[1])
    edges.append(((u_var, u_time), (v_var, v_time)))
    #model.add_edge((u_var, u_time), (v_var, v_time))

model.add_edges_from(edges)
const_bn = model.get_constant_bn()
const_bn.fit(flat_df, estimator="MLE")