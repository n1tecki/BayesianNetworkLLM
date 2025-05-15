"""
Experiment runner: compare VoI timeline vs. factual timeline
Author: 2025-05-14
"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
from pgmpy.inference import DBNInference

from src.VoI.voi_tools import simulate_voi_path
from tqdm import tqdm


# ------------------------------------------------------------------ #
# 1.  Helper – lead time
# ------------------------------------------------------------------ #
def _first_cross(tl, thr):
    for step in tl:
        if step["p_sepsis"] >= thr:
            return step["t"]
    return np.inf


def patient_lead_time(voi_tl, real_tl, thr=0.7):
    """
    Positive  -> VoI triggers earlier
    NaN       -> neither timeline crosses the threshold
    """
    tv = _first_cross(voi_tl,  thr)
    tr = _first_cross(real_tl, thr)
    return np.nan if np.isinf(tv) or np.isinf(tr) else tr - tv


# ------------------------------------------------------------------ #
# 2.  Robust patient grouping (index or column)
# ------------------------------------------------------------------ #
def _patient_groups(df: pd.DataFrame):
    if 'hadm_id' in df.columns:
        yield from df.groupby('hadm_id', sort=False)
    else:
        yield from df.groupby(level=0, sort=False)


# ------------------------------------------------------------------ #
# 3.  Main experiment
# ------------------------------------------------------------------ #
def run_experiment(df_test: pd.DataFrame,
                   inference: DBNInference,
                   lab_cols: List[str],
                   *,
                   conf_threshold: float = 0.6,
                   missing_bin: int | None = None,
                   min_gain: float = 1e-4          # <<< expose
                   ) -> pd.DataFrame:

    rows = []
    groups = list(_patient_groups(df_test))       # materialise once
    for hadm_id, pat_df in tqdm(groups,
                                total=len(groups),
                                desc="Processing stays"):
        # chronological order
        pat_df = pat_df.sort_values("timestamp")
        pat_df = (pat_df.reset_index(level=0, drop=True)
                  if isinstance(pat_df.index, pd.MultiIndex)
                  else pat_df.reset_index(drop=True))

        voi_tl = simulate_voi_path(
            pat_df, inference, lab_cols, hadm_id,
            conf_threshold=conf_threshold,
            missing_bin=missing_bin,
            allow_voi=True,
            #min_gain=min_gain
        )
        real_tl = simulate_voi_path(
            pat_df, inference, lab_cols, hadm_id,
            conf_threshold=conf_threshold,
            missing_bin=missing_bin,
            allow_voi=False,
            #min_gain=min_gain      # keeps signatures aligned
        )

        rows.append({
            "hadm_id": hadm_id,
            "voi_steps":  len(voi_tl),
            "voi_timeline": voi_tl,
            "real_steps": len(real_tl),
            "real_timeline": real_tl,
            "lead_time": patient_lead_time(voi_tl, real_tl, conf_threshold)
        })

    return pd.DataFrame(rows)
