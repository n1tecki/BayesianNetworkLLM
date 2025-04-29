import pandas as pd
from utils import export_table_to_csv, df_to_temporal
from sofa_classification import compute_sofa_scores, classify_sofa_stays
from preprocessing import clean_bloodgas
from stats import value_stats

# This script exports a table from the SQLite database to a CSV file
db_path = "sqlite_db/mimic4.db"
table_name = "_lab_chart_sofa_events"
output_csv = "data/_lab_chart_sofa_events.csv"
# export_table_to_csv(db_path, table_name, output_csv)

# Load the data and transform it into a temporal dataframe to CSV file
df = pd.read_csv('data/_lab_chart_sofa_events.csv')
labels_df = df[['hadm_id', 'sepsis']]
temporal_df = df_to_temporal(df)


# ------------------- DATA PREPROCESSING ------------------------------
wide_temporal_df = temporal_df.pivot_table(
    index=['hadm_id', 'timestamp'],
    columns='variable',
    values='value',
    aggfunc='first'
)
wide_temporal_df = clean_bloodgas(wide_temporal_df, fio2_col='FiO2', pao2_col='PaO2')


# ------------------- SOFA SORTING ------------------------------------
# Forward fill the lab values
all_sofa_vars = sorted(wide_temporal_df.columns.tolist())
wide_forward_temporal_df = wide_temporal_df.groupby(level=0)[all_sofa_vars].ffill()
# Compute the SOFA classification of each lab values
sofa_df = compute_sofa_scores(wide_forward_temporal_df)

# Classify from which timestap on sepsis was diagnosed, except stays with no sepsis diagnoses
sofa_df_classified = classify_sofa_stays(sofa_df, labels_df)

# Print example value
first_hadm = sofa_df_classified.index.unique(level=0)[1]
print(wide_forward_temporal_df.loc[first_hadm])
print(sofa_df_classified.loc[first_hadm])


# ------------------- STATISTICS --------------------------------------
# Summarise the count of lab values for a specific variable
all_sofa_vars = sorted(wide_temporal_df.columns.tolist())
lab_count_stats = value_stats(wide_temporal_df, all_sofa_vars)
for var, res in lab_count_stats.items():
    print(f"\n{var} counts:")
    print(res.counts)
    print(f"Admissions with {var}: {res.n_hadm}")

# Get stats aboiut the amount of timestamp events stays have
stay_lengths = sofa_df_classified.groupby('hadm_id').size()
dist = stay_lengths.value_counts().sort_index()
stay_timesteps_stats = (
    dist
    .rename_axis('rows_per_stay')
    .reset_index(name='num_stays')
)
print(stay_timesteps_stats)

# Print the amount of sepsis and non sepsis stays by diagnosis
stay_sepsis_flag = sofa_df_classified.groupby('hadm_id')['sepsis'].max()
counts = stay_sepsis_flag.value_counts().sort_index()
counts.index = ['no_sepsis', 'sepsis']
print(counts)

# Print the amount of sepsis and non sepsis stays by SEPSIS score
stay_sepsis_flag = sofa_df.groupby('hadm_id')['sepsis'].max()
counts = stay_sepsis_flag.value_counts().sort_index()
counts.index = ['no_sepsis', 'sepsis']
print(counts)