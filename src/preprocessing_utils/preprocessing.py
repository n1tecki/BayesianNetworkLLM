import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


# Transforms the raw SQL df data into a temporal dataframe
def df_to_temporal(df):
    df_local = df.copy()

    suffixes = (
        "_charttime",
        "_storetime",
        "_valuenum",
        "_value",
    )

    long_rows = []

    # 1️⃣ Work out every unique variable prefix (PaO2, platelet_count, …)
    prefixes = {
        c.rsplit("_", 1)[0]
        for c in df_local.columns
        if c.endswith(suffixes)
    }

    for prefix in prefixes:
        # 2️⃣ Pick the “best” timestamp column
        t_col = f"{prefix}_charttime" if f"{prefix}_charttime" in df_local.columns \
                else f"{prefix}_storetime"

        # 3️⃣ Pick the value columns in order of preference
        vnum_col  = f"{prefix}_valuenum" if f"{prefix}_valuenum" in df_local.columns else None
        vstr_col  = f"{prefix}_value"    if f"{prefix}_value"    in df_local.columns else None

        # 4️⃣ Slice out just the pieces we need
        sub = df_local[[c for c in ["hadm_id", t_col, vnum_col, vstr_col] if c]].copy()

        # 5️⃣ Prefer the numeric column; fall back to the string column
        if vnum_col:
            sub["value"] = sub[vnum_col]
        if vstr_col:
            sub["value"] = sub["value"].fillna(sub[vstr_col])

        # 6️⃣ Drop rows with no timestamp **or** no value
        sub = sub.dropna(subset=[t_col, "value"])
        sub = sub.rename(columns={t_col: "timestamp"})
        sub["variable"] = prefix

        long_rows.append(sub[["hadm_id", "timestamp", "variable", "value"]])

    # 7️⃣ Concatenate all variable blocks and order them
    tidy = (
        pd.concat(long_rows, ignore_index=True)
          .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
          .sort_values(["hadm_id", "timestamp", "variable"])
          .reset_index(drop=True)
    )

    tidy_temporal = tidy.pivot_table(
        index=['hadm_id', 'timestamp'],
        columns='variable',
        values='value',
        aggfunc='first'
    )

    return tidy_temporal


# Clean outlier or not standardsize bloodgas values
def clean_bloodgas(
    df: pd.DataFrame,
    fio2_col: str = 'FiO2',
    pao2_col: str = 'PaO2',
    max_fio2_percent: float = 100.0,
    min_fio2_percent: float = 15.0,
    min_pao2: float = 1.0,
    max_pao2: float = 500.0
) -> pd.DataFrame:
    df_local = df.copy()

    # --- Clean FiO2 ---
    fio2_raw = pd.to_numeric(df_local[fio2_col], errors='coerce')

    # Replace 0 and negative values with NaN
    fio2 = fio2_raw.mask(fio2_raw <= 0)

    # Convert fractions to percentages ONLY if <= 1.0 and > 0.1 (to avoid noise like 0.01 or 0.001)
    converted_fraction = fio2.between(0.1, 1.0)
    fio2.loc[converted_fraction] = fio2[converted_fraction] * 100

    # Remove values that are implausible: < min_fio2_percent or > max_fio2_percent
    fio2 = fio2.where((fio2 >= min_fio2_percent) & (fio2 <= max_fio2_percent))

    df_local[fio2_col] = fio2

    # --- Clean PaO2 ---
    pao2 = pd.to_numeric(df_local[pao2_col], errors='coerce').replace(0, np.nan)
    pao2 = pao2.where((pao2 >= min_pao2) & (pao2 <= max_pao2))

    df_local[pao2_col] = pao2

    return df_local



# Map gcs_motor string to numeric
def gcs_motor_to_numeric(df):
    df_local = df.copy()

    verbal_map = {
        'Normal': 5,
        'Slurred': 4,
        'Garbled': 3,
        'Aphasic': 2,
        'Mute': 1,
        'Intubated/trached': 1
    }
    df_local['gcs_motor'] = df_local['gcs_motor'].replace(verbal_map)
    return df_local


# Cleaning excessive outliers
def clean_min_max(df, column, min_val, max_val, replace=False):
    df_local = df.copy()
    orig = pd.to_numeric(df_local[column], errors='coerce')

    if replace:
        cleaned = orig.copy()
        cleaned = cleaned.mask(cleaned < 0, other=np.nan)
        cleaned = cleaned.mask(cleaned < min_val, other=min_val)
        cleaned = cleaned.mask(cleaned > max_val, other=max_val)
    else:
        mask = (orig < min_val) | (orig > max_val)
        cleaned = orig.mask(mask)

    # count how many entries actually changed (excluding NaN-to-NaN)
    changed = (orig != cleaned) & ~(orig.isna() & cleaned.isna())
    n_changed = changed.sum()
    print(f"For {column}-> {n_changed} values changed out of {len(orig)} total")

    df_local[column] = cleaned
    return df_local


# Forward fill the lab values
def forward_fill(df):
    df_local = df.copy()
    all_sofa_vars = sorted(df_local.columns.tolist())
    df_local = df_local.groupby(level=0)[all_sofa_vars].ffill()
    return df_local


# Adding the SOFA classifiactions to the raw value table
def adding_sofa_classification(df1, df2):
    return df1.join(df2['sepsis'], how='left')


def data_into_bins(df: pd.DataFrame, N_BINS) -> pd.DataFrame:

    LAB_COLS = [
        #"FiO2", "PaO2", 
        'pf_ratio',
        "bilirubin_total", "creatinin",
        "cns_score",
        #"gcs_eye", "gcs_motor", "gcs_verbal",
        "mean_arterial_pressure", "platelet_count",
    ]
    MISSING_BIN = N_BINS
    df_local = df.copy().reset_index()


    def qbin(col: str) -> None:
        ok = df_local[col].notna()
        kb = KBinsDiscretizer(N_BINS, encode="ordinal", strategy="quantile")
        df_local.loc[ok, col] = kb.fit_transform(df_local.loc[ok, [col]]).astype(int)
        df_local.loc[~ok, col] = MISSING_BIN
        df_local[col] = df_local[col].astype(int)

    for lab in LAB_COLS:
        qbin(lab)

    df_local["sepsis"] = df_local["sepsis"].astype(int)
    df_local['timestamp'] = pd.to_datetime(df_local['timestamp'])
    df_local = df_local.set_index("hadm_id").sort_values("timestamp")

    return df_local


def cns_transformation(df):
    df_local = df.copy()
    def cns_score(row):
        eye = row['gcs_eye'] if not np.isnan(row['gcs_eye']) else 4
        verbal = row['gcs_verbal'] if not np.isnan(row['gcs_verbal']) else 5
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


def compute_pf_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy()

    # 1) numeric coercion + fill FiO₂→21%
    df_local['FiO2'] = (
        pd.to_numeric(df_local.get('FiO2'), errors='coerce')
          .fillna(21.0)
    )

    # 2) numeric coercion for PaO₂ (no fill—let NaN propagate)
    df_local['PaO2'] = pd.to_numeric(df_local.get('PaO2'), errors='coerce')

    # 3) PF ratio
    df_local['pf_ratio'] = df_local['PaO2'] / (df_local['FiO2'] / 100)

    return df_local