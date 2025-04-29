from collections import namedtuple
import pandas as pd

# Summarise one or several variables in a pivoted wide DataFrame.
VarStats = namedtuple("VarStats", ["counts", "n_hadm"])

def value_stats(
    wide_df: pd.DataFrame,
    variables,
    *,
    include_nan: bool = False,
    force_numeric: bool | None = None,
):
    """
    Given a wide DataFrame indexed by (hadm_id, timestamp) with one column per variable,
    compute for each variable:
      - counts: value_counts (optionally including NaNs)
      - n_hadm: number of unique hadm_id with at least one non-null measurement
    
    Parameters
    ----------
    wide_df : pd.DataFrame
        Pivoted DataFrame whose index is a MultiIndex (hadm_id, timestamp).
    variables : str or list of str
        Column name(s) in wide_df to summarise.
    include_nan : bool
        If True, include NaN in the value_counts; else drop NaNs.
    force_numeric : bool or None
        - True: coerce all values to numeric (NaN if convertible fails).
        - False: keep values as-is.
        - None: auto-detect; if all non-missing values convert cleanly to numeric,
                cast to numeric; otherwise leave as-is.
    
    Returns
    -------
    dict
        Mapping var_name → VarStats(counts: pd.Series, n_hadm: int)
    """
    if isinstance(variables, str):
        variables = [variables]
    
    out = {}
    # extract level-0 of the MultiIndex (hadm_id)
    if isinstance(wide_df.index, pd.MultiIndex):
        hadm_level = wide_df.index.get_level_values(0)
    else:
        raise ValueError("wide_df must be indexed by a MultiIndex (hadm_id, timestamp)")
    
    for var in variables:
        if var not in wide_df.columns:
            raise KeyError(f"Variable '{var}' not found in wide_df.columns")
        
        # take the series of values for this var
        series = wide_df[var].copy()
        
        # decide whether to coerce to numeric
        if force_numeric is True:
            series = pd.to_numeric(series, errors="coerce")
        elif force_numeric is False:
            pass  # leave as-is
        else:
            # auto: if all non-null entries convert to numeric without NaN, cast
            nonnull = series.dropna()
            can_num = pd.to_numeric(nonnull, errors="coerce").notna().all()
            if can_num:
                series = pd.to_numeric(series, errors="coerce")
        
        # count values
        counts = series.value_counts(
            dropna = not include_nan
        ).sort_index()
        
        # number of unique hadm_id that have at least one non-null measurement
        nonnull_idx = series.dropna().index.get_level_values(0)
        n_hadm = pd.Index(nonnull_idx).nunique()
        
        out[var] = VarStats(counts=counts, n_hadm=n_hadm)
    
    return out
