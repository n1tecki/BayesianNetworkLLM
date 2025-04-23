import pandas as pd
import sqlite3

icd_iso10 = pd.read_csv("sqlite_db\CCS_codes\DXCCSR_v2025-1\DXCCSR_v2025-1.csv")
icd_iso10_codes = icd_iso10[icd_iso10["'CCSR CATEGORY 1 DESCRIPTION'"] == "Septicemia"][
    "'ICD-10-CM CODE'"
].tolist()
iso10_codes = [v.replace("'", "") for v in icd_iso10_codes]

icd_iso9 = pd.read_csv(
    "sqlite_db\CCS_codes\Single_Level_CCS_2015\$dxref 2015.csv", skiprows=1
)
icd_iso9_codes = icd_iso9[icd_iso9["'CCS CATEGORY DESCRIPTION'"] == "'Septicemia'"][
    "'ICD-9-CM CODE'"
].tolist()
iso9_codes = [v.replace("'", "") for v in icd_iso9_codes]


with open("sqlite_db\CCS_codes\sepsis_icd_codes.txt", "w", encoding="utf-8") as f:
    for item in iso10_codes + iso9_codes:
        f.write(item + "\n")


conn = sqlite3.connect("sqlite_db/mimic4.db")
cursor = conn.cursor()

# Create the table
cursor.execute("DROP TABLE IF EXISTS _sepsis_icd_codes")
cursor.execute("CREATE TABLE _sepsis_icd_codes (icd TEXT)")
cursor.executemany("INSERT INTO _sepsis_icd_codes (icd) VALUES (?)", [(code,) for code in iso10_codes + iso9_codes])

conn.commit()
conn.close()