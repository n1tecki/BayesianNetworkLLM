# -------------------------------------------------------------------------
# value_of_information.py
#
# Given:
#   * df_test      – your forward-filled test frame (MultiIndex: hadm_id, row)
#   * inference    – the DBNInference object returned by dbn_train()
#   * LAB_COLS     – list of laboratory variable names (same order everywhere)
#
# It returns:
#   info_dict[hadm_id] = [
#       {"step": k,
#        "lab": chosen_lab,
#        "gain": info_gain_k,
#        "p_post": p_sepsis_after_k}
#       ...
#   ]
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

EPS = 1e-12               # for numerical stability in entropy


def _binary_entropy(p: float) -> float:
    """H(p) with log₂; p is clipped into (EPS, 1-EPS)."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _first_nonzero(series: pd.Series) -> Any:
    """Return the first value that is neither 0 nor NaN, else None."""
    nz = series[(series != 0) & series.notna()]
    return nz.iloc[0] if not nz.empty else None


def simulate_patient(
    patient_df: pd.DataFrame,
    inference,
    LAB_COLS: List[str],
    thresh: float = 0.5,          # decision threshold for hard call
    use_entropy: bool = True      # True → information gain (entropy);
                                  # False → |Δ p(sepsis)| (“confidence jump”)
) -> List[Dict[str, Any]]:
    """
    Greedy VoI on slice 0 with *0 = un-measured* evidence in place.

    Returns a list of dicts:
        step, lab, gain,
        p_prev, p_post,
        p_no_sepsis, p_sepsis,
        decision   (1 = sepsis, 0 = no sepsis)
    """
    # earliest real measurement (first non-zero & non-NaN) for each lab
    earliest = {
        lab: _first_nonzero(patient_df[lab])
        for lab in LAB_COLS
    }
    candidate_labs = {lab for lab, v in earliest.items() if v is not None}

    # ---------------------------------------------------------------------
    # EVIDENCE starts with all labs = 0  (un-measured)
    # and we’ll overwrite entries as soon as that lab is “measured”.
    # ---------------------------------------------------------------------
    evidence = {(lab, 0): 0 for lab in LAB_COLS}

    results = []

    # ----- helper to get posterior p(sepsis=1) ----------------------------
    def posterior(ev_dict):
        q = inference.forward_inference(
            variables=[("sepsis", 0)], evidence=ev_dict
        )
        return float(q[("sepsis", 0)].values[1])

    # ----- baseline (step −1) --------------------------------------------
    p0 = posterior(evidence)
    results.append({
        "step": -1,
        "lab": None,
        "gain": 0.0,
        "p_prev": None,
        "p_post": p0,
        "p_no_sepsis": 1 - p0,
        "p_sepsis": p0,
        "decision": int(p0 >= thresh)
    })

    step = 0
    while candidate_labs:
        p_base = posterior(evidence)

        best_gain, best_lab, best_p = -np.inf, None, None
        best_abs = -np.inf                         # for |Δ p| when needed
        H_base = _binary_entropy(p_base) if use_entropy else None

        for lab in candidate_labs:
            test_ev = evidence.copy()
            test_ev[(lab, 0)] = earliest[lab]     # replace 0 by real value
            p = posterior(test_ev)

            if use_entropy:
                gain = H_base - _binary_entropy(p)    # information gain (≥ 0)
            else:
                gain = abs(p - p_base)                # “confidence jump”

            # tie-breaking consistent & deterministic
            better = (gain > best_gain + 1e-12) if use_entropy else \
                     (gain > best_abs  + 1e-12)
            if better:
                best_gain, best_lab, best_p = gain, lab, p
                if not use_entropy:
                    best_abs = gain

        if best_lab is None:            # numerical safety
            break

        # commit
        evidence[(best_lab, 0)] = earliest[best_lab]
        candidate_labs.remove(best_lab)

        results.append({
            "step": step,
            "lab": best_lab,
            "gain": best_gain,
            "p_prev": p_base,
            "p_post": best_p,
            "p_no_sepsis": 1 - best_p,
            "p_sepsis": best_p,
            "decision": int(best_p >= thresh)
        })
        step += 1

    return results




def value_of_information(
    df_test: pd.DataFrame,
    inference,
    LAB_COLS: List[str]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Wrapper for the whole cohort.
    Returns a dict keyed by hadm_id with the per-step lists.
    """
    info_dict: Dict[int, List[Dict[str, Any]]] = {}
    for hadm_id, patient_df in tqdm(df_test.groupby(level="hadm_id", sort=False)):
        info_dict[hadm_id] = simulate_patient(patient_df, inference, LAB_COLS)
    return info_dict
