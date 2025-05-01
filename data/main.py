import pandas as pd
from utils import export_table_to_csv
from sofa_classification import compute_sofa_scores, classify_sofa_stays, sofa_classification
from preprocessing import clean_bloodgas, gcs_motor_to_numeric, df_to_temporal, forward_fill, adding_sofa_classification, clean_min_max
from stats import lab_value_counts, count_leading_zeros_before_sepsis, sepsis_duration_count, sepsis_nonsepsis_count, plot_distribution_with_bell


# For the paper here clean the extreme values of each lab value. Report how many got caught and replaced.
# Then plot the distribution of the lab values.
# Also report on the  replacement og gcs_motoric names with the scores. 
# After that also report on the stats like amount of timestamp distribution.
# Then report also on the amount of false classification of sepsis when only taking the sofa score.


# This script exports a table from the SQLite database to a CSV file
# db_path = "sqlite_db/mimic4.db"
# table_name = "_lab_chart_sofa_events"
# output_csv = "data/_lab_chart_sofa_events.csv"
# export_table_to_csv(db_path, table_name, output_csv)


# Load the data and transform it into a temporal dataframe to CSV file
df = pd.read_csv('data/_lab_chart_sofa_events.csv')
labels_df = df[['hadm_id', 'sepsis']]


# ------------------- DATA PREPROCESSING ------------------------------
temporal_df = df_to_temporal(df)
raw_lab_count_stats = lab_value_counts(temporal_df)
temporal_df_clean = clean_bloodgas(temporal_df, fio2_col='FiO2', pao2_col='PaO2')
temporal_df_clean = gcs_motor_to_numeric(temporal_df_clean)
temporal_df_clean = clean_min_max(temporal_df_clean, column='bilirubin_total', min_val=.1, max_val=50, replace=False)
temporal_df_clean = clean_min_max(temporal_df_clean, column='creatinin', min_val=.2, max_val=20, replace=False)
temporal_df_clean = clean_min_max(temporal_df_clean, column='mean_arterial_pressure', min_val=20, max_val=200, replace=False)
temporal_df_clean = clean_min_max(temporal_df_clean, column='platelet_count', min_val=10, max_val=1000, replace=False)
clean_temporal_df_ff = forward_fill(temporal_df_clean)


# ------------------- SOFA SORTING ------------------------------------
sofa_df = compute_sofa_scores(clean_temporal_df_ff)
sofa_df_sofa_classification = sofa_classification(sofa_df)
sofa_df_diagnoses_classified = classify_sofa_stays(sofa_df_sofa_classification, labels_df)
clean_temporal_df_ff_sepsis = adding_sofa_classification(clean_temporal_df_ff, sofa_df_diagnoses_classified)


# ------------------- STATISTICS --------------------------------------
lab_count_stats = lab_value_counts(temporal_df_clean)
stay_timesteps_stats = sepsis_duration_count(sofa_df_diagnoses_classified)
sepsis_by_diagnoses = sepsis_nonsepsis_count(sofa_df_diagnoses_classified)
sepsis_by_sofa = sepsis_nonsepsis_count(sofa_df_diagnoses_classified)
duration_before_sofa = count_leading_zeros_before_sepsis(sofa_df_diagnoses_classified)


# Print example value
first_hadm = sofa_df_diagnoses_classified.index.unique(level=0)[1]
print(clean_temporal_df_ff_sepsis.loc[first_hadm])
print(sofa_df_diagnoses_classified.loc[first_hadm])

# plot_distribution_with_bell(raw_lab_count_stats['platelet_count'])
# plot_distribution_with_bell(raw_lab_count_stats['mean_arterial_pressure'])
# plot_distribution_with_bell(raw_lab_count_stats['creatinin'])
# plot_distribution_with_bell(raw_lab_count_stats['bilirubin_total'])
# plot_distribution_with_bell(raw_lab_count_stats['gcs_verbal'])
# plot_distribution_with_bell(raw_lab_count_stats['gcs_motor'])
# plot_distribution_with_bell(raw_lab_count_stats['gcs_eye'])
# plot_distribution_with_bell(raw_lab_count_stats['PaO2'])
# plot_distribution_with_bell(raw_lab_count_stats['FiO2'])

# plot_distribution_with_bell(lab_count_stats['platelet_count'])
# plot_distribution_with_bell(lab_count_stats['mean_arterial_pressure'])
# plot_distribution_with_bell(lab_count_stats['creatinin'])
# plot_distribution_with_bell(lab_count_stats['bilirubin_total'])
# plot_distribution_with_bell(lab_count_stats['gcs_verbal'])
# plot_distribution_with_bell(lab_count_stats['gcs_motor'])
# plot_distribution_with_bell(lab_count_stats['gcs_eye'])
# plot_distribution_with_bell(lab_count_stats['PaO2'])
# plot_distribution_with_bell(lab_count_stats['FiO2'])

clean_temporal_df_ff_sepsis.to_csv('data/raw_df_classified.csv', index=True, index_label=['hadm_id', 'timestamp'])
sofa_df_diagnoses_classified.to_csv('data/sofa_df_classified.csv', index=True, index_label=['hadm_id', 'timestamp'])
