from src.sql_utils.sql_script_exec import execute_sql_files
from src.sql_utils.utils import export_table_to_csv


# ------------------ SQL TABLE CREARION ------------------------------
sql_scripts = [
    "sqlite_db/query_scripts/diagnoses_icd_sepsis.sql",  # Selecting all patients and stays with sepsis diagnosis
    "sqlite_db/query_scripts/icustays_sepsis.sql", # Merging sepsis stays with admission information
    "sqlite_db/query_scripts/icustays_sepsis_filtered.sql", # Filter icu stays for minimum duration and no returning patients
    "sqlite_db/query_scripts/lab_chart_sofa_events.sql", # Selecting all lab events and chart events related to sofa for sepsis patients
    "sqlite_db/query_scripts/first_sepsis_action.sql", # Selecting all first sepsis actions for sepsis patients
]
execute_sql_files("sqlite_db/mimic4.db", sql_scripts)

db_path = "sqlite_db/mimic4.db"
table_name = "_lab_chart_sofa_events"
output_csv = "data/_lab_chart_sofa_events.csv"
export_table_to_csv(db_path, table_name, output_csv)