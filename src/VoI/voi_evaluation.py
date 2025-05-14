from src.VoI.voi_tools import simulate_voi_path
import pandas as pd
import numpy as np


def patient_lead_time(tl, real_tl, conf_threshold=0.7):
    """Difference (real - voi) in first time reaching threshold."""
    def first_cross(timeline):
        for step in timeline:
            if step['p_sepsis'] >= conf_threshold:
                return step['t']
        return np.inf
    return first_cross(real_tl) - first_cross(tl)

def run_experiment(df_test, inference, lab_cols,
                   conf_threshold=0.7):
    results = []
    for hadm_id, pat_df in df_test.groupby(level=0):
        pat_df = pat_df.droplevel(0)               # timestamp index only
        voi_tl  = simulate_voi_path(pat_df, inference, lab_cols,
                                    conf_threshold)
        real_tl = simulate_voi_path(pat_df, inference, lab_cols,      # same function
                                    conf_threshold=conf_threshold)    # but *no* extra labs
        lead = patient_lead_time(voi_tl, real_tl, conf_threshold)
        results.append({'hadm_id': hadm_id,
                        'voi_steps': len(voi_tl),
                        'real_steps': len(real_tl),
                        'lead_time': lead})
    return pd.DataFrame(results)
