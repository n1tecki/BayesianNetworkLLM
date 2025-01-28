import sqlite3
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseInitializer:
    """
    Class to initialize the SQLite database and create necessary tables.
    """
    
    def __init__(self, db_path="sqlite_db/patient_index.sqlite"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()

    def _connect(self):
        """Establish connection to SQLite database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logging.info("Connected to database successfully.")
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}")
            raise

    def execute_sql_file(self, file_path):
        """Executes SQL statements from a given file."""
        try:
            with open(file_path, 'r') as file:
                sql_script = file.read()
            self.cursor.executescript(sql_script)
            self.conn.commit()
            logging.info("SQL script executed successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error executing SQL script: {e}")
            raise
        except FileNotFoundError:
            logging.error("SQL file not found.")
            raise

    def close_connection(self):
        """Closes the database connection if it is open."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    db_initializer = DatabaseInitializer()
    db_initializer.execute_sql_file("sqlite_db/schema_setup.sql")
    db_initializer.close_connection()


# python sqlite_db\db_setup.py