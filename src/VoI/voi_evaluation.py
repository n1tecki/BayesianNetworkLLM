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
        if step['p_sepsis'] >= thr:
            return step['t']
    return np.inf


def patient_lead_time(voi_tl, real_tl, thr=0.7):
    """Positive ⇒ VoI detects earlier."""
    return _first_cross(real_tl, thr) - _first_cross(voi_tl, thr)


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
                   conf_threshold: float = 0.7,
                   missing_bin: int | None = None) -> pd.DataFrame:
    """
    Return a tidy DataFrame with columns:
      hadm_id | voi_steps | real_steps | lead_time
    """
    rows = []
    for hadm_id, pat_df in tqdm(_patient_groups(df_test), total=len(df_test), desc="Processing of all stays"):
        # ensure chronological order; drop hadm_id level if present
        pat_df = pat_df.sort_values('timestamp')
        if isinstance(pat_df.index, pd.MultiIndex):
            pat_df = pat_df.reset_index(level=0, drop=True)
        else:
            pat_df = pat_df.reset_index(drop=True)

        voi_tl = simulate_voi_path(
            pat_df, inference, lab_cols, hadm_id,
            conf_threshold=conf_threshold,
            missing_bin=missing_bin,
            allow_voi=True
        )
        real_tl = simulate_voi_path(
            pat_df, inference, lab_cols, hadm_id,
            conf_threshold=conf_threshold,
            missing_bin=missing_bin,
            allow_voi=False
        )

        rows.append({
            'hadm_id': hadm_id,
            'voi_steps': len(voi_tl),
            'voi_timeline': voi_tl,
            'real_steps': len(real_tl),
            'real_timeline': real_tl,
            'lead_time': patient_lead_time(voi_tl, real_tl, conf_threshold)
        })

    return pd.DataFrame(rows)
