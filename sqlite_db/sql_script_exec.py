import sqlite3


def execute_sql_files(db_path, sql_files):
    """
    Executes multiple SQL script files within a single SQLite connection.
    After executing all scripts, it prints the result of 'SELECT * FROM total_icd_patients LIMIT 10;'
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute all SQL scripts
        for sql_file in sql_files:
            with open(sql_file, "r", encoding="utf-8") as file:
                sql_script = file.read()
                cursor.executescript(sql_script)
                print(f"Executed: {sql_file}")

        conn.commit()  # Commit in case scripts modified the database
        print("\nAll SQL scripts executed successfully.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        conn.close()


sql_scripts = [
    # Filtering for diagnoses
    "sqlite_db/query_scripts/icd_aci.sql", # Create temporary table with icd codes of all aci relevant diagnoses
    "sqlite_db/query_scripts/icd_chf.sql", # Create temporary table with icd codes of all chf relevant diagnoses
    "sqlite_db/query_scripts/icd_sepsis.sql", # Create temporary table with icd codes of all sepsis relevant diagnoses
    "sqlite_db/query_scripts/icd_pneumonia.sql", # Create temporary table with icd codes of all pneumonia relevant diagnoses
    "sqlite_db/query_scripts/icd_gi.sql", # Create temporary table with icd codes of all gi relevant diagnoses
    "sqlite_db/query_scripts/icd_total.sql", # Creates tables for each group and then creates a UNION of all tables
    "sqlite_db/query_scripts/admissions_icd_total.sql", # Create a table with all icd relevant admissions data  

    # Filtering for lab tests
    "sqlite_db/query_scripts/itemid_bun.sql", # Create temporary table with icd codes of all bun relevant lab tests
    "sqlite_db/query_scripts/itemid_creatinine.sql", # Create temporary table with icd codes of all creatinine relevant lab tests
    "sqlite_db/query_scripts/itemid_hgb.sql", # Create temporary table with icd codes of all hgb relevant lab tests
    "sqlite_db/query_scripts/itemid_k.sql", # Create temporary table with icd codes of all k relevant lab tests
    "sqlite_db/query_scripts/itemid_lactate.sql", # Create temporary table with icd codes of all lactate relevant lab tests
    "sqlite_db/query_scripts/itemid_na.sql", # Create temporary table with icd codes of all na relevant lab tests
    "sqlite_db/query_scripts/itemid_platelets.sql", # Create temporary table with icd codes of all platelets relevant lab tests
    "sqlite_db/query_scripts/itemid_wbc.sql", # Create temporary table with icd codes of all wbc relevant lab tests
    "sqlite_db/query_scripts/itemid_total.sql", # Creates tables for each group and then creates a UNION of all tables
    "sqlite_db/query_scripts/labevents_itemid_total.sql", # Create a table with all lab test values and their patients

    "sqlite_db/query_scripts/matched_admission_labevents.sql" # Creating a table of matched admissions with icd diagnoses and lab events
    "sqlite_db/query_scripts/balanced_matched_admission_labevents.sql" # Balanse the table for all diagnoses categories to have the same occurence
    
]

execute_sql_files("sqlite_db/mimic4.db", sql_scripts)
