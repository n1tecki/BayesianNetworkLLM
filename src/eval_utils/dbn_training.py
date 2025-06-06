import pandas as pd
from itertools import permutations
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, MaximumLikelihoodEstimator
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import TabularCPD
import numpy as np
from tqdm import tqdm


def flatten_df(df, LAB_COLS):
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


def structure_learning(df, LAB_COLS, CORRELATION_THRESHOLD=0.4):
    # Select possibel edges
    slice1_cols = [f"{v}_1" for v in ["sepsis"] + LAB_COLS]
    corr_matrix   = df[slice1_cols].corr("spearman").abs()

    blacklist_edges = set()

    # 1) Never allow any lab → sepsis (at the same time slice)
    for t in (0, 1):
        for lab in LAB_COLS:
            blacklist_edges.add((f"{lab}_{t}", f"sepsis_{t}"))

    # 2) Never allow any lab at t=1 to point backwards to t=0
    for lab in LAB_COLS:
        for var in ["sepsis"] + LAB_COLS:
            blacklist_edges.add((f"{lab}_1", f"{var}_0"))

    for l1 in LAB_COLS:
        for l2 in LAB_COLS:
            if l1 != l2 and corr_matrix.loc[f"{l1}_1", f"{l2}_1"] < CORRELATION_THRESHOLD:
                # disallow l1_1 -> l2_1 *and* l2_1 -> l1_1
                blacklist_edges.add((f"{l1}_1", f"{l2}_1"))
                blacklist_edges.add((f"{l2}_1", f"{l1}_1"))

    # Structure Learning 
    structure_estimator = HillClimbSearch(df[slice1_cols])
    estimated_model = structure_estimator.estimate(
        scoring_method='bic-d',
        tabu_length=20,
        epsilon=0.0001,
        expert_knowledge=ExpertKnowledge(
        forbidden_edges=list(blacklist_edges)),
        #search_space=list(whitelist_edges))
    )

    return estimated_model


def make_missing_state_uninformative(model, labs, missing_state=0, t_slices=(0, 1)):
    """
    For every lab in `labs` and every time slice in `t_slices`
    force the CPD row that corresponds to `missing_state`
    to be identical for every configuration of the parents.
    That removes any predictive power from the '0 = not measured' code.
    """
    for lab in labs:
        for t in t_slices:
            cpd = model.get_cpds((lab, t))

            # current CPD values
            vals = cpd.get_values().copy()

            # --- 1. choose a constant row for the missing state ----------
            #
            # Here we take the *overall* mean of that row and
            # reuse it for every parent configuration, but
            # any constant works as long as it's the same
            # in every column.
            #
            row = missing_state
            constant = vals[row, :].mean()
            vals[row, :] = constant

            # --- 2. renormalise each column so it sums to 1 -------------
            col_sums = vals.sum(axis=0, keepdims=True)
            vals /= col_sums

            # --- 3. write the patched CPD back into the model ----------
            new_cpd = TabularCPD(
                variable=cpd.variable,
                variable_card=cpd.variable_card,
                values=vals,
                evidence=cpd.variables[1:] or None,
                evidence_card=list(cpd.cardinality[1:]) if cpd.variables[1:] else None,
                state_names=cpd.state_names,
            )
            model.remove_cpds(cpd)
            model.add_cpds(new_cpd)

    model.check_model()
    return model



def dbn_train(flat_df, LAB_COLS, CORRELATION_THRESHOLD = 0.4, alpha=1e-6):

    # Learning of the structure of the DBN
    estimated_model = structure_learning(flat_df, LAB_COLS, CORRELATION_THRESHOLD)

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
    model = make_missing_state_uninformative(model, LAB_COLS)
    inference = DBNInference(model)
    return model, inference




def predict_sepsis_stream(patient_df, inference, LAB_COLS):
    evidence_so_far, out = {}, {}
    for t, row in enumerate(patient_df[LAB_COLS].itertuples(index=False)):
        for i, lab in enumerate(LAB_COLS):
            evidence_so_far[(lab, t)] = row[i]          # append new data only
        q = inference.forward_inference([("sepsis", t)],
                                         evidence=evidence_so_far)
        out[t] = float(q[("sepsis", t)].values[1])
    return out


def dbn_predict(df, inference, LAB_COLS):
    predictions = {}
    test_ids = df.index.unique()
    for hadm_id in tqdm(test_ids):
        new_patient = df.loc[[hadm_id], LAB_COLS].reset_index(drop=True)
        pred = predict_sepsis_stream(new_patient, inference, LAB_COLS)
        predictions[hadm_id] = pred

    return predictions