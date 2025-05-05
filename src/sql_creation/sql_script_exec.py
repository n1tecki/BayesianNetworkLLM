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

        conn.commit()
        print("\nAll SQL scripts executed successfully.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        conn.close()

