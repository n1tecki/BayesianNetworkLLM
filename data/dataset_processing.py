import pandas as pd


# Load in raw sql retrieved data
df = pd.read_csv('data/balanced_matched_admission_labevents.csv')


# Chose lab values if missing
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
df['ref_range_lower'] = pd.to_numeric(df['ref_range_lower'], errors='coerce')
df['ref_range_upper'] = pd.to_numeric(df['ref_range_upper'], errors='coerce')
df['valuenum'] = df['valuenum'].fillna(df['value'])


# Classify the lab values into high, normal or low depending on the normal range
def classify_value_level(row):
    if row['valuenum'] < row['ref_range_lower']:
        return 'lower'
    elif row['ref_range_lower'] <= row['valuenum'] <= row['ref_range_upper']:
        return 'normal'
    else:
        return 'high'
df['value_level'] = df.apply(classify_value_level, axis=1)


# One-hot encode lab tests
columns_to_encode = ['lab_category']
columns_to_keep = ['subject_id', 'value_level', 'admittime', 'diagnoses_category']
df_encoded = pd.get_dummies(df[columns_to_encode + columns_to_keep], columns=columns_to_encode)


# Replace 1 -> row’s value_level; 0 -> 'false' in the lab test columns
for col in df_encoded.columns:
    if col.startswith('lab_category_'):  # these are your one-hot diagnosis columns
        df_encoded[col] = df_encoded.apply(
            lambda row: row['value_level'] if row[col] == 1 else 'false',
            axis=1
        )


# Merge row one-hot encodings where subject-id, admittime and diagnoses_category is similar
# than count the amount of lab value labels and select the highest
lab_category_columns = [col for col in df_encoded.columns if col.startswith("lab_category_")]
def most_frequent_except_false(col):
    counts = col.value_counts()
    counts = counts.drop('false', errors='ignore') # Remove 'false' from counts
    return counts.idxmax() if not counts.empty else 'false' # If counts are empty (only 'false' existed or all were empty), return 'false'
df_majority_counts = df_encoded.groupby(['subject_id', 'admittime', 'diagnoses_category'])[lab_category_columns].apply(lambda x: x.apply(most_frequent_except_false))
df_majority_counts_flat = df_majority_counts.reset_index()
df_majority_counts_flat = df_majority_counts_flat.ffill()


# One-hot encode diagnoses
columns_to_encode = ['diagnoses_category']
df_cleaned = pd.get_dummies(df_majority_counts_flat, columns=columns_to_encode)


# Export final dataset
df_cleaned.to_csv('data/cleaned_data.csv', index=False)