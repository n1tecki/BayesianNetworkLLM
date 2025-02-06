import sqlite3
import pandas as pd


def export_table_to_csv(db_path, table_name, output_csv):
    """
    Connects to the SQLite database, loads the specified table into a Pandas DataFrame,
    and saves it as a CSV file.
    """
    try:
        conn = sqlite3.connect(db_path)

        # Load the table into a DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)

        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Table '{table_name}' exported successfully to {output_csv}")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        conn.close()


db_path = "sqlite_db/mimic4.db"
table_name = "balanced_matched_admission_labevents"
output_csv = "data/balanced_matched_admission_labevents.csv"

export_table_to_csv(db_path, table_name, output_csv)
