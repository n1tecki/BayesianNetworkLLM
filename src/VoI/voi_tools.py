from pgmpy.inference import DBNInference
from scipy.stats import entropy
import numpy as np
import pandas as pd


# –– 1‑A.  Shannon‑entropy of P(sepsis=1|evidence) –– 
def sepsis_entropy(infer: DBNInference, evidence, t):
    q = infer.query(variables=[('sepsis', t)], evidence=evidence)
    p = q.values          # p[0] = P(sepsis=0), p[1] = P(sepsis=1)
    return entropy(p, base=2)

# –– 1‑B.  Expected Information Gain (EIG) of an *unobserved* lab X(t)
def expected_ig(infer: DBNInference, evidence, lab, t):
    """Returns ΔH = H(sepsis|evi) – E_L[H(sepsis|evi,L)]."""
    # prior entropy
    h_prior = sepsis_entropy(infer, evidence, t)

    # predictive distribution of the lab itself
    pred_lab = infer.query(variables=[(lab, t)], evidence=evidence)
    ig = h_prior
    for state_idx, p_lab_val in enumerate(pred_lab.values):
        # skip impossible values
        if p_lab_val == 0:
            continue
        new_ev = evidence.copy()
        new_ev[(lab, t)] = state_idx
        h_post = sepsis_entropy(infer, new_ev, t)
        ig -= p_lab_val * h_post
    return ig     # measured in bits



def choose_next_lab(infer, evidence, lab_cols, t, measured_mask,
                    min_gain_bits=0.01):
    """
    evidence:  {(var, time): state_int, ...}
    measured_mask: dict  {(lab, t): bool}   # True if we already have value
    """
    gains = {}
    for lab in lab_cols:
        key = (lab, t)
        if measured_mask.get(key, False):           # already observed → skip
            continue
        ig = expected_ig(infer, evidence, lab, t)
        gains[lab] = ig
    if not gains:
        return None, {}
    best_lab, best_gain = max(gains.items(), key=lambda kv: kv[1])
    if best_gain < min_gain_bits:       # nothing worth asking for
        return None, gains
    return best_lab, gains



def simulate_voi_path(patient_df: pd.DataFrame,
                      infer: DBNInference,
                      lab_cols,
                      conf_threshold=0.7):
    """
    patient_df –– one patient, original timeline, indexed by timestamp
    Returns a list of dicts per time‑step with posterior & chosen lab.
    """
    # we assume patient_df already *binned* as in your preprocessing
    t_max = len(patient_df) - 1
    measured_mask, evidence = {}, {}
    timeline = []

    for t, (_, row) in enumerate(patient_df.iterrows()):
        # --- incorporate everything that is already charted at time t ---
        for lab in lab_cols + ['sepsis']:         # include sepsis labels
            val = row[f"{lab}"]
            if val != np.nan:                     # binning already converts NaN→MISSING_BIN
                evidence[(lab, t)] = int(val)
                measured_mask[(lab, t)] = True

        # --- VoI step BEFORE looking at future measurements -------------
        next_lab, gains = choose_next_lab(infer, evidence, lab_cols, t,
                                          measured_mask)
        if next_lab is not None:
            # If the lab is ever measured for this patient, grab the *actual*
            # first value after (or at) t; otherwise fall back to 2nd best:
            future_vals = patient_df[next_lab].iloc[t:]
            real_val = future_vals[future_vals != np.nan].head(1)
            if not real_val.empty:
                evidence[(next_lab, t)] = int(real_val.iat[0])
                measured_mask[(next_lab, t)] = True
            else:
                # mark unavailable and let outer loop try again next step
                measured_mask[(next_lab, t)] = False

        # --- posterior after optional VoI lab ---------------------------
        post = infer.query(variables=[('sepsis', t)], evidence=evidence).values
        p_sepsis = post[1]

        timeline.append({
            't': t,
            'p_sepsis': p_sepsis,
            'entropy_bits': entropy(post, base=2),
            'voi_lab': next_lab,
            'gains': gains,
        })

        # optional early‑stop:
        if p_sepsis >= conf_threshold:
            break

    return timeline
