import pandas as pd
import numpy as np

def compute_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: tall DataFrame with columns [hadm_id, timestamp, variable, value].
    Output: wide DataFrame indexed by (hadm_id, timestamp) with columns
      [resp_score, coag_score, hepatic_score, cv_score,
       cns_score, renal_score, sofa_total].
    """
    # 1) Make a copy of the df
    wide = df.copy()

    # 2) Ensure numeric types for the labs/vitals
    num_cols = [
        'FiO2', 'PaO2', 'platelet_count',
        'bilirubin_total', 'creatinin',
        'mean_arterial_pressure',
        'gcs_eye', 'gcs_verbal'
    ]
    for col in num_cols:
        if col in wide:
            wide[col] = pd.to_numeric(wide[col], errors='coerce')

    # 3) Map gcs_motor string → numeric
    verbal_map = {
        'Normal': 5,
        'Slurred': 4,
        'Garbled': 3,
        'Aphasic': 2,
        'Mute': 1,
        'Intubated/trached': 1
    }
    wide['gcs_motor_score'] = wide.get('gcs_motor', pd.Series()).map(verbal_map)

    # 4) Compute PF ratio and ventilator flag
    wide['pf_ratio'] = wide['PaO2'] / (wide['FiO2'] / 100)
    wide['ventilated'] = wide.get('gcs_motor', '') == 'Intubated/trached'

    # 5) Define scoring functions
    def resp_score(r, vent):
        if pd.isna(r): 
            return np.nan
        if r >= 400: 
            return 0
        if r < 100 and vent: 
            return 4
        if r < 200 and vent: 
            return 3
        if r < 300: 
            return 2
        return 1

    def coag_score(p):
        if pd.isna(p): return np.nan
        if p >= 150: return 0
        if p >= 100: return 1
        if p >= 50:  return 2
        if p >= 20:  return 3
        return 4

    def hepatic_score(b):
        if pd.isna(b): return np.nan
        if b < 1.2:  return 0
        if b < 2.0:  return 1
        if b < 6.0:  return 2
        if b < 12.0: return 3
        return 4

    def cv_score(map_):
        if pd.isna(map_): return np.nan
        return 0 if map_ >= 70 else 1
        # (for full vasoactive scoring you’d need drug‐rates)

    def cns_score(row):

        eye   = row['gcs_eye'] if not np.isnan(row['gcs_eye']) else 4
        verbal= row['gcs_verbal'] if not np.isnan(row['gcs_verbal']) else 5
        motor = row['gcs_motor_score'] if not np.isnan(row['gcs_motor_score']) else 6
        total_gcs = eye + verbal + motor

        if pd.isna(total_gcs): return np.nan
        if total_gcs == 15:   return 0
        if total_gcs >= 13:   return 1
        if total_gcs >= 10:   return 2
        if total_gcs >= 6:    return 3
        return 4

    def renal_score(cr):
        if pd.isna(cr): return np.nan
        if cr < 1.2:  return 0
        if cr < 2.0:  return 1
        if cr < 3.5:  return 2
        if cr < 5.0:  return 3
        return 4

    # 6) Apply them
    wide['resp_score'] = wide.apply(
        lambda row: resp_score(row['pf_ratio'], row['ventilated']), axis=1
    )
    wide['coag_score']    = wide['platelet_count'].apply(coag_score)
    wide['hepatic_score'] = wide['bilirubin_total'].apply(hepatic_score)
    wide['cv_score']      = wide['mean_arterial_pressure'].apply(cv_score)

    # Combine the three GCS subcomponents into total GCS, then score CNS
    wide['cns_score'] = wide.apply(cns_score, axis=1)

    wide['renal_score'] = wide['creatinin'].apply(renal_score)

    # 7) Sort by hadm_id, then time, forward‐fill & default‐0, then sum
    comp_cols = [
        'resp_score', 'coag_score', 'hepatic_score',
        'cv_score', 'cns_score', 'renal_score'
    ]

    wide = wide.sort_index(level=['hadm_id', 'timestamp'])
    # wide[comp_cols] = (
    #     wide
    #     .groupby(level=0)[comp_cols]
    #     .ffill()        # carry last known score forward
    #     .fillna(0)      # if never measured, assume 0
    # )

    wide['sofa_total'] = wide[comp_cols].sum(axis=1)

    # 8) Return only the scores, still indexed by (hadm_id, timestamp)
    return wide[comp_cols + ['sofa_total']]


def classify_sofa_stays(sofa_df, labels_df):
    df = sofa_df.copy()
    # Classify from which timestap on sepsis was diagnosed, except stays with no sepsis diagnoses
    df['sepsis'] = (df['sofa_total'] >= 2).astype(int)

    # Build a dict mapping hadm_id → stay‐level sepsis label
    label_map = labels_df.set_index('hadm_id')['sepsis'].to_dict()

    # Get an array of stay‐level labels aligned to each sofa_df row
    stay_labels = df.index.get_level_values('hadm_id').map(label_map)

    # Wherever the stay label == 0, force sub‐row sepsis to 0
    df.loc[stay_labels == 0, 'sepsis'] = 0 

    return df
