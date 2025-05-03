from pathlib import Path
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def data_into_bins(df: pd.DataFrame, N_BINS) -> pd.DataFrame:

    LAB_COLS = [
        "FiO2", "PaO2", "bilirubin_total", "creatinin",
        "gcs_eye", "gcs_motor", "gcs_verbal",
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

    print("✓  binned_train_data.parquet written")
    return df_local
