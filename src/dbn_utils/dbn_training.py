import pandas as pd
from itertools import permutations
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, MaximumLikelihoodEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import TabularCPD
import numpy as np
from tqdm import tqdm


def flatten_df(df, LAB_COLS):
    # — Configuration ——————————————————————————————————————————
    df = df.groupby(level="hadm_id").filter(lambda group: len(group) > 1)

    # Flatten dataframe into two time slices (t0 -> t1)
    def two_slice(df, labs):
        base = ["sepsis"] + labs
        d = df.reset_index().sort_values(["hadm_id", "timestamp"])
        now  = d[base].add_suffix("_1")
        prev = d.groupby("hadm_id")[base].shift(1).add_suffix("_0")
        return pd.concat([prev, now], axis=1).dropna().reset_index(drop=True)
    flat_df = two_slice(df, LAB_COLS)

    return flat_df



def dbn_train(flat_df, LAB_COLS, CORRELATION_THRESHOLD = 0.4, alpha=1e-6):
    # — Select possibel edges —————————————————————————————————
    slice1_cols = [f"{v}_1" for v in ["sepsis"] + LAB_COLS]
    corr_matrix = flat_df[slice1_cols].corr("spearman").abs()
    
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
    all_possible_edges = set(permutations(slice1_cols, 2))
    blacklist_edges = all_possible_edges - whitelist_edges


    # — Structure Learning ———————————————————————————————————————
    structure_estimator = HillClimbSearch(flat_df[slice1_cols])
    estimated_model = structure_estimator.estimate(
        scoring_method='bic-d',
        expert_knowledge=ExpertKnowledge(
        forbidden_edges=list(blacklist_edges),
        search_space=list(whitelist_edges))
    )

    # keep only edges WITH child in slice 1  (child endswith '_1')
    # in_slice_edges = [(u, v) for u, v in estimated_model.edges() if v.endswith("_1")]


    # — DBN Model Creation ———————————————————————————————————————
    model = DBN()

    edges = []
    # 1)  persistence edges  var(0) -> var(1)
    #for var in ["sepsis"] + LAB_COLS: is it the right direction?
    for var in ["sepsis"] + LAB_COLS:
        edges.append(((var, 0), (var, 1)))
    # 2)  learned in-slice edges  u(1) -> v(1)
    for u, v in estimated_model.edges():
        u_var, u_time = u.rsplit('_', 1)
        v_var, v_time = v.rsplit('_', 1)
        edges.append(((u_var, int(u_time)), (v_var, int(v_time))))
        #model.add_edge((u_var, u_time), (v_var, v_time))

    model.add_edges_from(edges)
    const_bn = model.get_constant_bn()
    const_bn.fit(flat_df, estimator=MaximumLikelihoodEstimator)
    node_map = {
        (n.node, n.time_slice): n
        for n in model.nodes()
    }

    # — CPD Creation —————————————————————————————————————————————
    # Split them into three categories
    initial_cpds     = {}  # P(X_0) -- no evidence
    intra_slice_cpds = {}  # P(X_0 | …) -- evidence also at time 0
    transition_cpds  = {}  # P(X_1 | …) -- evidence from time 0

    for cpd in const_bn.cpds:
        var, t_str = cpd.variable.rsplit('_', 1)
        t = int(t_str)

        if t == 0 and not cpd.get_evidence():
            initial_cpds[var] = cpd
        elif t == 0 and cpd.get_evidence():
            intra_slice_cpds[var] = cpd
        elif t == 1:
            transition_cpds[var] = cpd

    def relabel_cpd_as_tuple(cpd):
        var, t = cpd.variable.rsplit("_", 1)
        t = int(t)

        ev = [(e.rsplit("_", 1)[0], int(e.rsplit("_", 1)[1]))
            for e in cpd.variables[1:]] or None
        ev_card = list(cpd.cardinality[1:]) if ev else None
        s_names = {(n.rsplit("_", 1)[0], int(n.rsplit("_", 1)[1])): s
                for n, s in cpd.state_names.items()}
        
        vals = np.asarray(cpd.get_values(), dtype=np.float64)
        vals += alpha
        vals /= vals.sum(axis=0, keepdims=True)

        return TabularCPD(
            variable      =(var, t),
            variable_card = cpd.variable_card,
            values        = vals,
            evidence      = ev,
            evidence_card = ev_card,
            state_names   = s_names,
        )

    for cpd in initial_cpds.values():
        model.add_cpds(relabel_cpd_as_tuple(cpd))
    for cpd in intra_slice_cpds.values():
        model.add_cpds(relabel_cpd_as_tuple(cpd))
    for cpd in transition_cpds.values():
        model.add_cpds(relabel_cpd_as_tuple(cpd))

    model.initialize_initial_state()
    assert model.check_model()
    inference = DBNInference(model)
    return model, inference


def predict_sepsis(patient_df, inference, LAB_COLS):
    out = {}

    for t, row in patient_df.reset_index(drop=True).iterrows():
        # only evidence from *this* slice – no gigantic dict:
        evidence_slice = {(lab, t): row[lab] for lab in LAB_COLS}

        q = inference.forward_inference(
            variables=[("sepsis", t)],
            evidence=evidence_slice
        )
        out[t] = q[("sepsis", t)].values[1]

        # If you want to keep a running filter, you can
        # evidence_so_far.update(evidence_slice)
    return out


def dbn_predict(df, inference, LAB_COLS):
    predictions = {}
    test_ids = df.index.unique()
    for hadm_id in tqdm(test_ids):
        new_patient = df.loc[[hadm_id], LAB_COLS].reset_index(drop=True)
        pred = predict_sepsis(new_patient, inference, LAB_COLS)
        predictions[hadm_id] = pred

    return predictions