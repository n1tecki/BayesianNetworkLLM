import pandas as pd
from src.preprocessing.sofa_classification import compute_sofa_scores, classify_sofa_stays, sofa_classification, adding_sepsis_classification_per_stay, cns_transformation
from src.preprocessing.preprocessing import clean_bloodgas, gcs_motor_to_numeric, df_to_temporal, forward_fill, adding_sofa_classification, clean_min_max
from src.preprocessing.stats import lab_value_counts, count_leading_zeros_before_sepsis, sepsis_duration_count, sepsis_nonsepsis_count, plot_distribution_with_bell
from src.preprocessing.data_binning import data_into_bins


# For the paper here clean the extreme values of each lab value. Report how many got caught and replaced.
# Then plot the distribution of the lab values.
# Also report on the  replacement og gcs_motoric names with the scores. 
# After that also report on the stats like amount of timestamp distribution.
# Then report also on the amount of false classification of sepsis when only taking the sofa score.
# Add that during training the spesis scores are set to priors like .8/.2 not hard 1/0 in order to not overfit the model / is a moderate nudge.


# Load the data and transform it into a temporal dataframe to CSV file
df = pd.read_csv('data/_lab_chart_sofa_events.csv')
labels_df = df[['hadm_id', 'sepsis']].reset_index()


# ------------------- DATA PREPROCESSING ------------------------------
temporal_df = df_to_temporal(df)
raw_lab_count_stats = lab_value_counts(temporal_df)
temporal_df_clean = clean_bloodgas(temporal_df, fio2_col='FiO2', pao2_col='PaO2')
temporal_df_clean = gcs_motor_to_numeric(temporal_df_clean)
temporal_df_clean = clean_min_max(temporal_df_clean, column='bilirubin_total', min_val=.1, max_val=50, replace=False)
temporal_df_clean = clean_min_max(temporal_df_clean, column='creatinin', min_val=.2, max_val=20, replace=False)
temporal_df_clean = clean_min_max(temporal_df_clean, column='mean_arterial_pressure', min_val=30, max_val=200, replace=False)
temporal_df_clean = clean_min_max(temporal_df_clean, column='platelet_count', min_val=10, max_val=1000, replace=False)
temporal_df_clean = cns_transformation(temporal_df_clean)
clean_temporal_df_ff = forward_fill(temporal_df_clean)


# ------------------- SOFA SORTING ------------------------------------
sofa_df = compute_sofa_scores(clean_temporal_df_ff)
sofa_df_sofa_classification = sofa_classification(sofa_df)
sofa_df_diagnoses_classified = classify_sofa_stays(sofa_df_sofa_classification, labels_df)
clean_temporal_df_ff_supervised = adding_sofa_classification(clean_temporal_df_ff, sofa_df_diagnoses_classified)
clean_temporal_df_ff_semisupervised = adding_sepsis_classification_per_stay(clean_temporal_df_ff, labels_df)
clean_temporal_df_ff_unsupervised = clean_temporal_df_ff.copy()
binned_train_data = data_into_bins(clean_temporal_df_ff_semisupervised, N_BINS=3)


# ------------------- STATISTICS --------------------------------------
lab_count_stats = lab_value_counts(temporal_df_clean)
stay_timesteps_stats = sepsis_duration_count(sofa_df_diagnoses_classified)
sepsis_by_diagnoses = sepsis_nonsepsis_count(sofa_df_diagnoses_classified)
sepsis_by_sofa = sepsis_nonsepsis_count(sofa_df_diagnoses_classified)
duration_before_sofa = count_leading_zeros_before_sepsis(sofa_df_diagnoses_classified)


# Graphs and stats ----------------------------------------------------
first_hadm = sofa_df_diagnoses_classified.index.unique(level=0)[1]
print(clean_temporal_df_ff_supervised.loc[first_hadm])
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


# plot_distribution_with_bell(lab_count_stats['platelet_count'], unit=r"($\times 10^9$/L)", label="Platelet Count", binsize=.25)
# plot_distribution_with_bell(lab_count_stats['mean_arterial_pressure'], unit="mmhg", label="Mean Arterial Pressure ", binsize=.25)
# plot_distribution_with_bell(lab_count_stats['creatinin'], unit="mg/dL", label="Creatinin", binsize=.25)
# plot_distribution_with_bell(lab_count_stats['bilirubin_total'], unit="mg/dL", label="Total Bilirubin", binsize=.25)
# plot_distribution_with_bell(lab_count_stats['gcs_verbal'], unit=r"", label="CNS Verbal Score", binsize=1)
# plot_distribution_with_bell(lab_count_stats['gcs_motor'], unit=r"", label="CNS Motor Score", binsize=1)
# plot_distribution_with_bell(lab_count_stats['gcs_eye'], unit=r"", label="CNS Eye Score", binsize=1)
# plot_distribution_with_bell(lab_count_stats['PaO2'], label=r"$\mathrm{PaO_2}$ - Fraction of Inspired Oxygen", binsize=8)
# plot_distribution_with_bell(lab_count_stats['FiO2'], label=r"$\mathrm{FiO_2}$ - Fraction of Inspired Oxygen", binsize=8)
# plot_distribution_with_bell(lab_count_stats['cns_score'], unit=r"", label="Central nervous system Score", binsize=1)


clean_temporal_df_ff_unsupervised.to_parquet('data/unsupervised_df_classified.parquet', engine='pyarrow')
clean_temporal_df_ff_semisupervised.to_parquet('data/semisupervised_df_classified.parquet', engine='pyarrow')
clean_temporal_df_ff_supervised.to_parquet('data/supervisedraw_df_classified.parquet', engine='pyarrow')
sofa_df_diagnoses_classified.to_parquet('data/sofa_df_classified.parquet', engine='pyarrow')
binned_train_data.to_parquet('data/binned_train_data.parquet', engine='pyarrow')