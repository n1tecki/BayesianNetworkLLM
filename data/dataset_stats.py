import pandas as pd
from utils import export_table_to_csv, df_to_temporal, value_stats

# This script exports a table from the SQLite database to a CSV file
db_path = "sqlite_db/mimic4.db"
table_name = "_lab_chart_sofa_events"
output_csv = "data/_lab_chart_sofa_events.csv"
export_table_to_csv(db_path, table_name, output_csv)

# Load the data and transform it into a temporal dataframe to CSV file
df = pd.read_csv('data/_lab_chart_sofa_events.csv')
temporal_df = df_to_temporal(df)

# Summarise the data for a specific variable
all_sofa_variables = sorted(temporal_df["variable"].unique().tolist())
sofa_variables = ["gcs_motor", "gcs_eye"]
multi = value_stats(temporal_df, sofa_variables)
for var, res in multi.items():
    print(f"\n{var} counts:")
    print(res.counts)
    print(f"Admissions with {var}: {res.n_hadm}")