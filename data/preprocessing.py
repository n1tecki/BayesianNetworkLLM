import pandas as pd
import numpy as np

def clean_bloodgas(
    df: pd.DataFrame,
    fio2_col: str = 'FiO2',
    pao2_col: str = 'PaO2',
    max_fio2_percent: float = 100.0,
    min_pao2: float = 1.0,
    max_pao2: float = 500.0
) -> pd.DataFrame:
    """
    Clean up inspired O2 (FiO2) and arterial O2 (PaO2) in your tall DataFrame.

    For FiO2:
      - Zero or non-numeric → NaN
      - Values <= 1.0 → treated as fraction → scaled to percent (×100)
      - Values > 1.0 → assumed percent → left as-is
      - Values > max_fio2_percent → set to NaN

    For PaO2:
      - Zero or non-numeric → NaN
      - Values < min_pao2 or > max_pao2 → set to NaN

    Parameters
    ----------
    df : pd.DataFrame
        Input “tall” DataFrame with columns at least [hadm_id, timestamp, variable, value].
    fio2_col : str
        Name of the FiO2 measurement variable in your pivoted DataFrame.
    pao2_col : str
        Name of the PaO2 measurement variable.
    max_fio2_percent : float
        Maximum physiologic FiO2 (%) allowed. Anything above becomes NaN.
    min_pao2 : float
        Minimum physiologic PaO2 (mmHg) allowed. Anything below becomes NaN.
    max_pao2 : float
        Maximum physiologic PaO2 (mmHg) allowed. Anything above becomes NaN.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with the FiO2 and PaO2 columns cleaned in-place.
    """
    df = df.copy()

    # --- Clean FiO2 ---
    # coerce to numeric, zeros → NaN
    fio2 = pd.to_numeric(df[fio2_col], errors='coerce').replace(0, np.nan)

    # fraction → percent, percent stays
    fio2_clean = np.where(
        fio2 <= 1.0,
        fio2 * 100,
        fio2
    )
    # clip impossible highs
    fio2_clean = np.where(
        fio2_clean > max_fio2_percent,
        np.nan,
        fio2_clean
    )
    df[fio2_col] = fio2_clean

    # --- Clean PaO2 ---
    # coerce to numeric, zeros → NaN
    pao2 = pd.to_numeric(df[pao2_col], errors='coerce').replace(0, np.nan)

    # mask out-of-range values
    pao2_clean = pao2.where(
        (pao2 >= min_pao2) & (pao2 <= max_pao2),
        np.nan
    )
    df[pao2_col] = pao2_clean

    return df
