import pandas as pd
from itertools import permutations
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

# -------------------------------------------------------------------- #
#  CONFIG
# -------------------------------------------------------------------- #
LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
CORR_TH = 0.5

# -------------------------------------------------------------------- #
#  LOAD & FLATTEN DATA  (t-1  →  t)
# -------------------------------------------------------------------- #
df = pd.read_parquet("data/binned_train_data.parquet")
df = df.groupby(level="hadm_id").filter(lambda g: len(g) > 1)

def two_slice(df, labs):
    base = ["sepsis"] + labs
    d = df.reset_index().sort_values(["hadm_id", "timestamp"])
    now  = d[base].add_suffix("_1")
    prev = d.groupby("hadm_id")[base].shift(1).add_suffix("_0")
    return pd.concat([prev, now], axis=1).dropna().reset_index(drop=True)

flat_df = two_slice(df, LAB_COLS)

# -------------------------------------------------------------------- #
#  LEARN IN-SLICE (t = 1) STRUCTURE ONLY
# -------------------------------------------------------------------- #
slice1_cols = [f"{v}_1" for v in ["sepsis"] + LAB_COLS]
corr = flat_df[slice1_cols].corr("spearman").abs()

wlist = set()
# sepsis(1) -> every lab(1)
wlist |= {("sepsis_1", f"{lab}_1") for lab in LAB_COLS}
# lab-lab edges at t=1 with ρ ≥ threshold
for l1 in LAB_COLS:
    for l2 in LAB_COLS:
        if l1 != l2 and corr.loc[f"{l1}_1", f"{l2}_1"] >= CORR_TH:
            wlist.add((f"{l1}_1", f"{l2}_1"))

# black-list everything else among slice-1 vars
all_pairs = set(permutations(slice1_cols, 2))
blist = all_pairs - wlist

hc  = HillClimbSearch(flat_df[slice1_cols])
dag = hc.estimate(scoring_method=BicScore(flat_df[slice1_cols]),
                  white_list=list(wlist),
                  black_list=list(blist))

# keep only edges WITH child in slice 1  (child endswith '_1')
in_slice_edges = [(u, v) for u, v in dag.edges() if v.endswith("_1")]

# -------------------------------------------------------------------- #
#  BUILD DBN GRAPH  (no edge has child at slice 0!)
# -------------------------------------------------------------------- #
model = DBN()

# 1)  persistence edges  var(0) -> var(1)
for var in ["sepsis"] + LAB_COLS:
    model.add_edge((var, 0), (var, 1))

# 2)  learned in-slice edges  u(1) -> v(1)
for u, v in in_slice_edges:
    u_var = u.replace("_1", "")
    v_var = v.replace("_1", "")
    model.add_edge((u_var, 1), (v_var, 1))

# -------------------------------------------------------------------- #
#  PARAMETER LEARNING  via constant BN
# -------------------------------------------------------------------- #
const_bn = model.get_constant_bn()        # names like "PaO2_0", "PaO2_1"
const_bn.fit(flat_df, estimator=MaximumLikelihoodEstimator)

# copy CPDs into DBN
for cpd in const_bn.cpds:
    var, t = cpd.variable.rsplit("_", 1)
    t       = int(t)

    ev = [(e.rsplit("_", 1)[0], int(e.rsplit("_", 1)[1]))
          for e in cpd.variables[1:]] or None
    ev_card = list(cpd.cardinality[1:]) if ev else None
    s_names = {(n.rsplit("_", 1)[0], int(n.rsplit("_", 1)[1])): s
               for n, s in cpd.state_names.items()}

    model.add_cpds(TabularCPD(
        variable      =(var, t),
        variable_card = cpd.variable_card,
        values        = cpd.get_values(),
        evidence      = ev,
        evidence_card = ev_card,
        state_names   = s_names,
    ))

# -------------------------------------------------------------------- #
#  FINALISE DBN   (this is where it used to crash)
# -------------------------------------------------------------------- #
model.initialize_initial_state()   # ✅  now passes
assert model.check_model()

# -------------------------------------------------------------------- #
#  INFERENCE
# -------------------------------------------------------------------- #
infer = DBNInference(model)

infer = DBNInference(model)

def predict_sepsis(patient_df):
    out = {}
    evidence_so_far = {}

    for t, row in patient_df.reset_index(drop=True).iterrows():
        # only evidence from *this* slice – no gigantic dict:
        evidence_slice = {(lab, t): row[lab] for lab in LAB_COLS}

        q = infer.forward_inference(
            variables=[("sepsis", t)],
            evidence=evidence_slice
        )
        out[t] = q[("sepsis", t)].values[1]

        # If you want to keep a running filter, you can
        # evidence_so_far.update(evidence_slice)
    return out


# ---- example ------------------------------------------------------- #
hadm_id = 29280967
new_patient = df.loc[hadm_id].reset_index(drop=True)[LAB_COLS]
print(predict_sepsis(new_patient))




for lab in LAB_COLS:
    cpd = model.get_cpds((lab, 1))
    print(f"{lab!r} slice 1 accepts states: {cpd.state_names[(lab,1)]}")