from collections import namedtuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Summarise one or several variables in a pivoted wide DataFrame.
def lab_value_counts(df, include_nan=False, force_numeric=None):
    out = {}
    for col in sorted(df.columns):
        s = df[col]

        # numeric coercion
        if force_numeric is True:
            s = pd.to_numeric(s, errors="coerce")
        elif force_numeric is None:
            nonnull = s.dropna()
            nums = pd.to_numeric(nonnull, errors="coerce")
            if nums.notna().all():
                s = nums

        counts = s.value_counts(dropna=not include_nan).sort_index()
        n_hadm = s.dropna().index.get_level_values(0).nunique()

        out[col] = (counts, n_hadm)
        print(f"\n{col} counts:")
        print(counts)
        print(f"Admissions with {col}: {n_hadm}")

    return out



# Counts how many timestamps before sepsis sofa score reached
def count_leading_zeros_before_sepsis(df: pd.DataFrame) -> pd.Series:
    df_local = df.copy()

    # 1. Identify which rows belong to stays that ever hit sepsis==1
    stay_max = df_local.groupby(level='hadm_id')['sepsis'].transform('max')
    df_pos = df_local[stay_max == 1]

    # 2. Helper: count zeros before the first 1 in a Series
    def _count_zeros_before_first_one(series: pd.Series) -> int:
        arr = series.values
        # find indices where sepsis==1
        ones_idx = np.where(arr == 1)[0]
        # if for some reason there's no 1, return NaN (shouldn't happen here)
        if ones_idx.size == 0:
            return np.nan
        first_one = ones_idx[0]
        # count zeros before that
        return int(np.sum(arr[:first_one] == 0))

    # 3. Apply per-stay and tabulate
    zeros_per_stay = (
        df_pos
        .groupby(level='hadm_id')['sepsis']
        .apply(_count_zeros_before_first_one)
    )
    distribution = zeros_per_stay.value_counts().sort_index()
    print(distribution)

    return distribution


# Get stats about the amount of timestamp events stays have
def sepsis_duration_count(df):
    df_local = df.copy()
    stay_lengths = df_local.groupby('hadm_id').size()
    dist = stay_lengths.value_counts().sort_index()
    stay_timesteps_stats = (
        dist
        .rename_axis('rows_per_stay')
        .reset_index(name='num_stays')
    )
    print(stay_timesteps_stats)

    return stay_timesteps_stats


# Print the amount of sepsis and non sepsis stays
def sepsis_nonsepsis_count(df):
    df_local = df.copy()
    stay_sepsis_flag = df_local.groupby('hadm_id')['sepsis'].max()
    counts = stay_sepsis_flag.value_counts().sort_index()
    counts.index = ['no_sepsis', 'sepsis']
    print(counts)

    return counts


def plot_distribution_with_bell(data, unit, label='empty', binsize=0.5):
    """
    Plot histogram (density) of data and overlay a full normal bell curve
    extending to mean ± 3 std, even beyond the observed data range.
    """
    # 1) Normalize to counts
    if isinstance(data, tuple):
        first, second = data
        if isinstance(first, pd.Series):
            counts = first.sort_index()
        else:
            vals, cnts = first, second
            counts = pd.Series(cnts, index=vals).sort_index()
    elif isinstance(data, pd.Series):
        counts = data.sort_index()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # 2) Expand to observations
    obs = np.repeat(
        counts.index.values.astype(float),
        counts.values.astype(int)
    )

    # 3) Histogram bin edges
    obs_min, obs_max = obs.min(), obs.max()
    bins = np.arange(obs_min, obs_max + binsize, binsize)

    # 4) Plot histogram as density
    plt.figure(figsize=(8, 4))
    plt.hist(obs, bins=bins, density=True, edgecolor='black')

    # 5) Compute mean and std
    mu, sigma = np.mean(obs), np.std(obs)

    # 6) Full bell range: mean ± 3·std
    x_min, x_max = mu - 3*sigma, mu + 3*sigma
    x = np.linspace(x_min, x_max, 1000)
    
    # 7) Normal PDF
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))
    plt.plot(x, pdf)

    # 8) Enforce full bell limits
    plt.xlim(x_min, x_max)

    # 9) Labels
    plt.xlabel(f'Value in {unit}')
    plt.ylabel('Density')
    plt.title(f'Distribution of Laboratory Value {label}')
    plt.tight_layout()
    plt.show()
