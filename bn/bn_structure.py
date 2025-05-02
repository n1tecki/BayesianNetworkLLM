

import pandas as pd
from itertools import permutations
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN, BayesianModel
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import TabularCPD

# — your settings — 
LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
RHO_CUT = 0.3  # correlation threshold

df = pd.read_parquet("data/binned_train_data.parquet")

# — 0) Drop stays that only have one record —
df = df.groupby(level="hadm_id").filter(lambda grp: len(grp) > 1)

# — 1) Build the 2-slice “flat” DataFrame —
def create_flattened(df, lab_cols):
    vars0 = ["sepsis"] + lab_cols
    df2   = df.reset_index().sort_values(["hadm_id", "timestamp"])
    df1   = df2[vars0].copy()
    df1.columns = [f"{v}_1" for v in vars0]
    df0   = df2.groupby("hadm_id")[vars0].shift(1)
    df0.columns = [f"{v}_0" for v in vars0]
    flat  = pd.concat([df0, df1], axis=1).dropna().reset_index(drop=True)
    return flat

flat = create_flattened(df, LAB_COLS)

# — 2) Compute Spearman correlations for lab–lab edges —
rho = flat[[f"{c}_1" for c in ["sepsis"] + LAB_COLS]].corr(method="spearman").abs()

# — 3) Build white/black lists —
white = set()
for v in ["sepsis"] + LAB_COLS:
    white.add((f"{v}_0", f"{v}_1"))
for lab in LAB_COLS:
    white.add(("sepsis_1", f"{lab}_1"))
for i in LAB_COLS:
    for j in LAB_COLS:
        if i != j and rho.loc[f"{i}_1", f"{j}_1"] >= RHO_CUT:
            white.add((f"{i}_1", f"{j}_1"))

all_vars  = list(flat.columns)
all_edges = set(permutations(all_vars, 2))
forbidden = all_edges - white

# — 4) Structure learning (2-slice DAG) —
hc    = HillClimbSearch(flat)
model = hc.estimate(
    scoring_method=BicScore(flat),
    white_list=list(white),
    black_list=list(forbidden),
)
print("Learned edges:", model.edges())

# — 5) Build the DBN skeleton —
dbn = DBN()
for v in ["sepsis"] + LAB_COLS:
    dbn.add_node(v)
for u, v in model.edges():
    src, ts = u.rsplit("_", 1)
    dst, td = v.rsplit("_", 1)
    dbn.add_edge((src, int(ts)), (dst, int(td)))
print("DBN structure edges:", dbn.edges())

# — 6) Parameter learning via constant-BN hack —
const_bn = dbn.get_constant_bn()
const_bn.fit(flat, estimator=MaximumLikelihoodEstimator)

for cpd in const_bn.get_cpds():
    var, t = cpd.variable.rsplit("_", 1)
    var_node = (var, int(t))
    evidence = []
    evidence_card = []
    if cpd.evidence:
        for ev, card in zip(cpd.evidence, cpd.evidence_card):
            name, ts = ev.rsplit("_", 1)
            evidence.append((name, int(ts)))
            evidence_card.append(card)

    dyn_cpd = TabularCPD(
        variable=var_node,
        variable_card=cpd.variable_card,
        values=cpd.get_values(),
        evidence=evidence or None,
        evidence_card=evidence_card or None,
        state_names=cpd.state_names
    )
    dbn.add_cpds(dyn_cpd)

# — 7) Inference setup —
infer = DBNInference(dbn)

def predict_sepsis_over_time(new_labs_df):
    evidence = {}
    probs    = {}
    for t, row in new_labs_df.iterrows():
        for lab in LAB_COLS:
            evidence[(lab, t)] = row[lab]
        q = infer.query(variables=[("sepsis", t)], evidence=evidence)
        probs[t] = q[("sepsis", t)].values[1]
    return probs

# — example usage —
# new_df = pd.DataFrame({ ... }, index=[0,1,2])
# print(predict_sepsis_over_time(new_df))
