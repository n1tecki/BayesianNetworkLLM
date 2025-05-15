# ───────────── src/VoI/voi_tools.py ─────────────
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

from tqdm import tqdm
import numpy as np
import pandas as pd
from pgmpy.inference import DBNInference
from pgmpy.factors.discrete import DiscreteFactor
from typing import Set


# ------------------------------------------------------------------ #
# 0.  Wrapper: unify pgmpy 0.x  vs  ≥1.0  query output
# ------------------------------------------------------------------ #
def _query_factor(infer: DBNInference, variables, evidence):
    """
    Always return the DiscreteFactor object (not the dict) that represents
    P(variables | evidence), regardless of pgmpy version.
    """
    out = infer.query(variables=variables, evidence=evidence)
    if isinstance(out, dict):                 # pgmpy ≥ 1.0
        return next(iter(out.values()))
    return out                                # pgmpy 0.x  already factor


# ------------------------------------------------------------------ #
# 1.  Convert Factor → 1-D float array  (no giant swiss-army knife!)
# ------------------------------------------------------------------ #
def _factor_to_array(f: DiscreteFactor | np.ndarray) -> np.ndarray:
    """
    pgmpy DiscreteFactor   →  ndarray
    ndarray                →  ndarray (copy as float)
    """
    if isinstance(f, np.ndarray):
        return f.astype(float).ravel()

    if isinstance(f, DiscreteFactor):
        vals = f.values if not callable(f.values) else f.values()
        return np.asarray(vals, dtype=float).ravel()

    raise TypeError(f"Expected DiscreteFactor or ndarray, got {type(f)}")


# ------------------------------------------------------------------ #
# 2.  Small helpers
# ------------------------------------------------------------------ #
def _not_missing(v: Any, miss_bin: Optional[int]) -> bool:
    return not (pd.isna(v) or (miss_bin is not None and v == miss_bin))


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ------------------------------------------------------------------ #
# 3.  Entropy & Expected Information Gain
# ------------------------------------------------------------------ #
def sepsis_entropy(infer: DBNInference,
                   evidence: Dict[Tuple[str, int], int],
                   t: int) -> float:
    key = ("sepsis", t)
    if key in evidence:
        return 0.0
    fac = _query_factor(infer, [key], evidence)
    return _entropy_bits(_factor_to_array(fac))


def expected_ig(infer: DBNInference,
                evidence: Dict[Tuple[str, int], int],
                lab: str,
                t: int) -> float:
    h0 = sepsis_entropy(infer, evidence, t)
    fac_lab = _query_factor(infer, [(lab, t)], evidence)
    lab_probs = _factor_to_array(fac_lab)

    ig = h0
    for state, p_lab in enumerate(lab_probs):
        if p_lab == 0:
            continue
        ev2 = evidence.copy()
        ev2[(lab, t)] = state
        ig -= p_lab * sepsis_entropy(infer, ev2, t)
    return ig


# ------------------------------------------------------------------ #
# 4.  Choose next lab
# ------------------------------------------------------------------ #
def choose_next_lab(infer: DBNInference,
                    evidence: Dict[Tuple[str, int], int],
                    available_labs: List[str],
                    t: int,
                    min_gain: float = 1e-6        # << very small
                    ) -> Tuple[Optional[str], Dict[str, float]]:
    gains = {lab: expected_ig(infer, evidence, lab, t) for lab in available_labs}
    if not gains:
        return None, {}
    best_lab, best_gain = max(gains.items(), key=lambda kv: kv[1])
    return (best_lab, gains) if best_gain > min_gain else (None, gains)



# ------------------------------------------------------------------ #
# 5.  Simulate a single patient trajectory
# ------------------------------------------------------------------ #
def simulate_voi_path(df_pat: pd.DataFrame,
                      infer: DBNInference,
                      lab_cols: List[str],
                      hadm_id,
                      *,
                      conf_threshold: float = 0.7,
                      missing_bin: Optional[int] = None,
                      allow_voi: bool = True) -> List[dict]:
    """
    1. Add ALL non‐missing (forward‐filled) labs each t.
    2. From among UNORDERED labs, pick the one with highest EIG *once*.
    3. Mark it ordered and pull its value (t or first non‐missing in future).
    4. Compute P(sepsis_t) | current evidence.
    5. Repeat until threshold or end.
    """
    df_pat = df_pat.sort_values("timestamp").reset_index(drop=True)

    evidence: Dict[Tuple[str, int], int] = {}
    ordered: Set[str] = set()      # only labs VoI has requested
    timeline: List[dict] = []

    for t, row in df_pat.iterrows():
        # ── A) Add dataset values (forward-filled) ───────────────────────
        for lab in lab_cols:
            v = row[lab]
            if _not_missing(v, missing_bin):
                evidence[(lab, t)] = int(v)

        # ── B) Greedy VoI ordering ───────────────────────────────────────
        voi_lab, gain = None, None
        if allow_voi:
            candidates = [lab for lab in lab_cols if lab not in ordered]
            if candidates:
                # compute all gains
                gains = {lab: expected_ig(infer, evidence, lab, t)
                         for lab in candidates}
                # pick best
                voi_lab = max(gains, key=gains.get)
                gain    = gains[voi_lab]
                ordered.add(voi_lab)

                # attach its measurement immediately (t or next non-missing)
                v0 = row[voi_lab]
                if _not_missing(v0, missing_bin):
                    evidence[(voi_lab, t)] = int(v0)
                else:
                    fut = df_pat.loc[t:, voi_lab]
                    nxt = fut[fut.apply(_not_missing, args=(missing_bin,))]
                    if not nxt.empty:
                        evidence[(voi_lab, t)] = int(nxt.iloc[0])

        # ── C) Posterior on sepsis_t ────────────────────────────────────
        fac   = _query_factor(infer, [("sepsis", t)], evidence)
        probs = _factor_to_array(fac)

        timeline.append({
            "t": t,
            "p_sepsis":   float(probs[1]),
            "entropy_bits": _entropy_bits(probs),
            "voi_lab":    voi_lab,   # the one we just ordered (or None)
            "gain":       gain       # its EIG
        })

        if probs[1] >= conf_threshold:
            break

    return timeline