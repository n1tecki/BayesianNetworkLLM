import sqlite3


# Populate SQLite database
conn = sqlite3.connect("my_database.db")
cursor = conn.cursor()

# Insert records using Python lists
data = [
    ("Alice", 25, "New York", 60000),
    ("Bob", 30, "Los Angeles", 70000),
    ("Charlie", 35, "Chicago", 80000)
]

cursor.executemany("INSERT INTO my_table (name, age, city, salary) VALUES (?, ?, ?, ?)", data)

conn.commit()
conn.close()

print("Data inserted successfully.")