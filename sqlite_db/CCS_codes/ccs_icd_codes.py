import pandas as pd
import sqlite3

# Read ICD-10
icd_iso10 = pd.read_csv("sqlite_db/CCS_codes/DXCCSR_v2025-1/DXCCSR_v2025-1.csv", low_memory=False)
icd_iso10_codes = icd_iso10[
    icd_iso10["'CCSR CATEGORY 1 DESCRIPTION'"].isin(["Septicemia", "Septicemia (except in labor)"])
][["'ICD-10-CM CODE'", "'ICD-10-CM CODE DESCRIPTION'"]].values.tolist()

# Read ICD-9
icd_iso9 = pd.read_csv(
    "sqlite_db/CCS_codes/Single_Level_CCS_2015/$dxref 2015.csv", skiprows=1
)
icd_iso9_codes = icd_iso9[
    icd_iso9["'CCS CATEGORY DESCRIPTION'"].isin(["'Septicemia'", "'Septicemia (except in labor)'"])
][["'ICD-9-CM CODE'", "'ICD-9-CM CODE DESCRIPTION'"]].values.tolist()

# Clean up quotes (remove ' around codes and descriptions)
iso10_codes_clean = [(code.replace("'", ""), name.replace("'", "")) for code, name in icd_iso10_codes]
iso9_codes_clean = [(code.replace("'", ""), name.replace("'", "")) for code, name in icd_iso9_codes]

# Save to text file
with open("sqlite_db/CCS_codes/sepsis_icd_codes.txt", "w", encoding="utf-8") as f:
    for code, diagnosis_description in iso10_codes_clean + iso9_codes_clean:
        f.write(f"{code}\t{diagnosis_description}\n")  # write tab-separated

# Insert into SQLite
conn = sqlite3.connect("sqlite_db/mimic4.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS _sepsis_icd_codes")
cursor.execute("CREATE TABLE _sepsis_icd_codes (icd TEXT, diagnosis_description TEXT)")

cursor.executemany(
    "INSERT INTO _sepsis_icd_codes (icd, diagnosis_description) VALUES (?, ?)",
    iso10_codes_clean + iso9_codes_clean
)

conn.commit()
conn.close()
