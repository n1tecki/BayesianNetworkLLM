import pandas as pd
import sqlite3

# Connects to the SQLite database, loads the specified table into a Pandas DataFrame
def export_table_to_csv(db_path, table_name, output_csv):
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

# Transforms the raw SQL df data into a temporal dataframe
def df_to_temporal(df):
    suffixes = (
        "_charttime",
        "_storetime",
        "_valuenum",
        "_value",
    )

    long_rows = []

    # 1️⃣ Work out every unique variable prefix (PaO2, platelet_count, …)
    prefixes = {
        c.rsplit("_", 1)[0]
        for c in df.columns
        if c.endswith(suffixes)
    }

    for prefix in prefixes:
        # 2️⃣ Pick the “best” timestamp column
        t_col = f"{prefix}_charttime" if f"{prefix}_charttime" in df.columns \
                else f"{prefix}_storetime"

        # 3️⃣ Pick the value columns in order of preference
        vnum_col  = f"{prefix}_valuenum" if f"{prefix}_valuenum" in df.columns else None
        vstr_col  = f"{prefix}_value"    if f"{prefix}_value"    in df.columns else None

        # 4️⃣ Slice out just the pieces we need
        sub = df[[c for c in ["hadm_id", t_col, vnum_col, vstr_col] if c]].copy()

        # 5️⃣ Prefer the numeric column; fall back to the string column
        if vnum_col:
            sub["value"] = sub[vnum_col]
        if vstr_col:
            sub["value"] = sub["value"].fillna(sub[vstr_col])

        # 6️⃣ Drop rows with no timestamp **or** no value
        sub = sub.dropna(subset=[t_col, "value"])
        sub = sub.rename(columns={t_col: "timestamp"})
        sub["variable"] = prefix

        long_rows.append(sub[["hadm_id", "timestamp", "variable", "value"]])

    # 7️⃣ Concatenate all variable blocks and order them
    tidy = (
        pd.concat(long_rows, ignore_index=True)
          .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
          .sort_values(["hadm_id", "timestamp", "variable"])
          .reset_index(drop=True)
    )

    return tidy


