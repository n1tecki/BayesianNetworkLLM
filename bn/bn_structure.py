import pandas as pd
from itertools import permutations
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import TabularCPD
import numpy as np

# — Configuration — 
LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
CORRELATION_THRESHOLD = 0.4

# Load and preprocess data
df = pd.read_parquet("data/binned_train_data.parquet")

# Filter out patients with only one observation
df = df.groupby(level="hadm_id").filter(lambda group: len(group) > 1)
# 6 entries with only 1 timestamp removed

# Flatten dataframe into two time slices (t0 -> t1)
def create_time_slices(df, lab_cols):
    current_vars = ["sepsis"] + lab_cols
    df_sorted = df.reset_index().sort_values(["hadm_id", "timestamp"])

    df_current = df_sorted[current_vars].copy()
    df_current.columns = [f"{var}_1" for var in current_vars]

    df_prev = df_sorted.groupby("hadm_id")[current_vars].shift(1)
    df_prev.columns = [f"{var}_0" for var in current_vars]

    flat_df = pd.concat([df_prev, df_current], axis=1).dropna().reset_index(drop=True)
    return flat_df

flat_df = create_time_slices(df, LAB_COLS)


# Edge Selection -------------------------------------------------------------------------------------------------
# Compute Spearman correlation
corr_matrix = flat_df[[f"{col}_1" for col in ["sepsis"] + LAB_COLS]].corr(method="spearman").abs()

# Define allowed edges (whitelist)
whitelist_edges = set()

# Edges between each lab t0 and each lab t1
for var in ["sepsis"] + LAB_COLS:
    whitelist_edges.add((f"{var}_0", f"{var}_1"))

# Edges between sepsis and all value in t1
for lab in LAB_COLS:
    whitelist_edges.add(("sepsis_1", f"{lab}_1"))

for lab1 in LAB_COLS:
    for lab2 in LAB_COLS:
        print(lab1, lab2)
        print(corr_matrix.loc[f"{lab1}_1", f"{lab2}_1"])
        if lab1 != lab2 and corr_matrix.loc[f"{lab1}_1", f"{lab2}_1"] >= CORRELATION_THRESHOLD:
            whitelist_edges.add((f"{lab1}_1", f"{lab2}_1"))

# Define forbidden edges (blacklist)
all_vars = list(flat_df.columns)
all_possible_edges = set(permutations(all_vars, 2))
blacklist_edges = all_possible_edges - whitelist_edges



# Structure Creation -----------------------------------------------------------------------------------------------
# Structure Learning
structure_estimator = HillClimbSearch(flat_df)
estimated_model = structure_estimator.estimate(
    scoring_method=BicScore(flat_df),
    white_list=list(whitelist_edges),
    black_list=list(blacklist_edges),
)

print(estimated_model.edges())


# Create Dynamic Bayesian Network
model = DBN()
model.add_edge([('sepsis_0', 'sepsis_1'), ('FiO2_0', 'FiO2_1'), ('PaO2_0', 'PaO2_1'), ('bilirubin_total_0', 'bilirubin_total_1'), ('creatinin_0', 'creatinin_1'), ('gcs_eye_0', 'gcs_eye_1'), ('gcs_motor_0', 'gcs_motor_1'), ('gcs_verbal_0', 'gcs_verbal_1'), ('mean_arterial_pressure_0', 'mean_arterial_pressure_1'), ('platelet_count_0', 'platelet_count_1'), ('sepsis_1', 'PaO2_1'), ('sepsis_1', 'creatinin_1'), ('sepsis_1', 'mean_arterial_pressure_1'), ('sepsis_1', 'gcs_motor_1'), ('sepsis_1', 'platelet_count_1'), ('creatinin_1', 'bilirubin_total_1'), ('gcs_verbal_1', 'gcs_eye_1')])




# Add nodes and edges dynamically
# for var in ["sepsis"] + LAB_COLS:
#    model.add_nodes_from([(var, 0), (var, 1)])

for source, target in estimated_model.edges():
    src_var, src_time = source.rsplit('_', 1)
    tgt_var, tgt_time = target.rsplit('_', 1)

    src = (src_var, int(src_time))
    tgt = (tgt_var, int(tgt_time))

    if src not in model:
        model.add_node(src)
    if tgt not in model:
        model.add_node(tgt)

    model.add_edge(src, tgt)

# Parameter Learning
const_bn = model.get_constant_bn()
const_bn.fit(flat_df, estimator=MaximumLikelihoodEstimator)
assert model.check_model()

# Transfer CPDs to DBN
for cpd in const_bn.cpds:
    var, t = cpd.variable.rsplit("_", 1)
    time_slice = int(t)

    # translate parent names to tuple form
    parents = cpd.variables[1:]
    evidence = [(p.rsplit("_", 1)[0], int(p.rsplit("_", 1)[1])) for p in parents]

    # parent cardinalities (needed only if there are parents)
    evidence_card   = list(cpd.cardinality[1:]) if evidence else None

     # rebuild state_names so keys match DBN tuple-variables ----
    new_state_names = {}
    for full_name, states in cpd.state_names.items():
        v, ts = full_name.rsplit("_", 1)
        new_state_names[(v, int(ts))] = states

    # construct the time-indexed CPD
    dynamic_cpd = TabularCPD(
        variable       =(var, time_slice),
        variable_card  = cpd.variable_card,
        values         = cpd.get_values(),
        evidence       = evidence or None,
        evidence_card  = evidence_card,
        state_names    = new_state_names
    )
    model.add_cpds(dynamic_cpd)

# Initialize inference engine
inference = DBNInference(model)

# Predict sepsis over multiple timesteps
def predict_sepsis(df_timesteps):
    evidence, predictions = {}, {}
    for timestep, row in df_timesteps.iterrows():
        evidence.update({(lab, timestep): row[lab] for lab in LAB_COLS})
        query_result = inference.query(variables=[("sepsis", timestep)], evidence=evidence)
        predictions[timestep] = query_result[("sepsis", timestep)].values[1]
    return predictions

# Example usage:
new_patient_data = pd.DataFrame(df.loc[29280967], index=[range(0, len(df.loc[29280967])-1)])
predictions = predict_sepsis(new_patient_data)
print(predictions)
