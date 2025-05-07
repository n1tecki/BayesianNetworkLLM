import pandas as pd
import numpy as np


# Classify trhe lab values to the SOFA classification
def compute_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Make a copy of the df
    df_local = df.copy()

    # 2) Ensure numeric types for the labs/vitals
    num_cols = [
        'FiO2', 'PaO2', 'platelet_count',
        'bilirubin_total', 'creatinin',
        'mean_arterial_pressure',
        'gcs_eye', 'gcs_verbal'
    ]
    for col in num_cols:
        if col in df_local:
            df_local[col] = pd.to_numeric(df_local[col], errors='coerce')

    # 3) Compute PF ratio and ventilator flag
    df_local['pf_ratio'] = df_local['PaO2'] / (df_local['FiO2'] / 100)
    df_local['ventilated'] = df_local.get('gcs_motor', '') == 'Intubated/trached'

    # 4) Define scoring functions
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
        motor = row['gcs_motor'] if not np.isnan(row['gcs_motor']) else 6
        total_gcs = eye + verbal + motor

        if pd.isna(total_gcs): return np.nan
        if total_gcs >= 15:   return 0
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

    # 5) Apply them
    df_local['resp_score'] = df_local.apply(
        lambda row: resp_score(row['pf_ratio'], row['ventilated']), axis=1
    )
    df_local['coag_score']    = df_local['platelet_count'].apply(coag_score)
    df_local['hepatic_score'] = df_local['bilirubin_total'].apply(hepatic_score)
    df_local['cv_score']      = df_local['mean_arterial_pressure'].apply(cv_score)

    # Combine the three GCS subcomponents into total GCS, then score CNS
    df_local['cns_score'] = df_local.apply(cns_score, axis=1)

    df_local['renal_score'] = df_local['creatinin'].apply(renal_score)

    # 6) Sort by hadm_id, then time, forward‐fill & default‐0, then sum
    comp_cols = [
        'resp_score', 'coag_score', 'hepatic_score',
        'cv_score', 'cns_score', 'renal_score'
    ]

    df_local = df_local.sort_index(level=['hadm_id', 'timestamp'])

    df_local['sofa_total'] = df_local[comp_cols].sum(axis=1)

    # 7) Return only the scores, still indexed by (hadm_id, timestamp)
    return df_local[comp_cols + ['sofa_total']]


# Classify from which timestap on sepsis was diagnosed, except stays with no sepsis diagnoses
def classify_sofa_stays(df, labels_df):
    df_local = df.copy()

    # Build a dict mapping hadm_id → stay‐level sepsis label
    label_map = labels_df.set_index('hadm_id')['sepsis'].to_dict()

    # Get an array of stay‐level labels aligned to each df_local row
    stay_labels = df_local.index.get_level_values('hadm_id').map(label_map)

    # Wherever the stay label == 0, force sub‐row sepsis to 0
    df_local.loc[stay_labels == 0, 'sepsis'] = 0 

    return df_local


# Classify from which timestap on sepsis was diagnosed, taking SOFA classification
def sofa_classification(df):
    df_local = df.copy()
    df_local['sepsis'] = (df_local['sofa_total'] >= 2).astype(int)

    return df_local


# Add the SEPSIS classification to each stay only
def adding_sepsis_classification_per_stay(df, labels_df):
    df_local = df.copy()
    labels_df_local = labels_df.copy()

    df_local = df_local.reset_index()
    df_local = df_local.merge(labels_df_local[['hadm_id', 'sepsis']], on='hadm_id', how='left')
    df_local = df_local.set_index(['hadm_id', 'sepsis', 'timestamp'])
    return df_local.sort_index()


def cns_transformation(df):
    df_local = df.copy()
    def cns_score(row):
        eye   = row['gcs_eye'] if not np.isnan(row['gcs_eye']) else 4
        verbal= row['gcs_verbal'] if not np.isnan(row['gcs_verbal']) else 5
        motor = row['gcs_motor'] if not np.isnan(row['gcs_motor']) else 6
        total_gcs = eye + verbal + motor
        return total_gcs

        if pd.isna(total_gcs): return np.nan
        if total_gcs >= 15:   return 0
        if total_gcs >= 13:   return 1
        if total_gcs >= 10:   return 2
        if total_gcs >= 6:    return 3
        return 4
    
    df_local['cns_score'] = df_local.apply(cns_score, axis=1)
    return df_local
    