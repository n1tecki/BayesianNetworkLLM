"""
Discretise the raw SOFA-lab time-series.
Writes: data/binned_train_data.parquet
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

RAW_PARQUET = Path("data/semisupervised_df_classified.parquet")
OUT_PARQUET = Path("data/binned_train_data.parquet")

LAB_COLS = [
    "FiO2", "PaO2", "bilirubin_total", "creatinin",
    "gcs_eye", "gcs_motor", "gcs_verbal",
    "mean_arterial_pressure", "platelet_count",
]
N_BINS = 5
MISSING_BIN = N_BINS

df = (
    pd.read_parquet(RAW_PARQUET)
      .reset_index()
      .sort_values(["hadm_id", "timestamp"])
)

def qbin(col: str) -> None:
    ok = df[col].notna()
    kb = KBinsDiscretizer(N_BINS, encode="ordinal", strategy="quantile")
    df.loc[ok, col] = kb.fit_transform(df.loc[ok, [col]]).astype(int)
    df.loc[~ok, col] = MISSING_BIN
    df[col] = df[col].astype(int)

for lab in LAB_COLS:
    qbin(lab)

df["sepsis"] = df["sepsis"].astype(int)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index("hadm_id").sort_values("timestamp")

Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET)

print("✓  binned_train_data.parquet written")
