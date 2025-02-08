import pandas as pd
import numpy as np
from missforest import MissForest



# Load in raw sql retrieved data
print('Loading data...')
df = pd.read_csv('data/balanced_matched_admission_labevents.csv')



# Chose lab values if missing
print('Cleaning data...')
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
df['ref_range_lower'] = pd.to_numeric(df['ref_range_lower'], errors='coerce')
df['ref_range_upper'] = pd.to_numeric(df['ref_range_upper'], errors='coerce')
df['valuenum'] = df['valuenum'].fillna(df['value'])



# Classify the lab values into high, normal or low depending on the normal range
print('Classifying data...')
def classify_value_level(row):
    if row['valuenum'] < row['ref_range_lower']:
        return 'low'
    elif row['ref_range_lower'] <= row['valuenum'] <= row['ref_range_upper']:
        return 'normal'
    elif row['valuenum'] > row['ref_range_upper']:
        return 'high'
df['value_level'] = df.apply(classify_value_level, axis=1)



# One-hot encode lab tests
print('One-hot encoding lab tests...')
columns_to_encode = ['lab_category']
columns_to_keep = ['subject_id', 'value_level', 'admittime', 'diagnoses_category']
df_encoded = pd.get_dummies(df[columns_to_encode + columns_to_keep], columns=columns_to_encode)



# Replace 1 -> row’s value_level; 0 -> False in the lab test columns
print('Replacing 1 -> row’s value_level; 0 -> false in the lab test columns...')
for col in df_encoded.columns:
    if col.startswith('lab_category_'):  # these are your one-hot diagnosis columns
        df_encoded[col] = df_encoded.apply(
            lambda row: row['value_level'] if row[col] == 1 else False,
            axis=1
        )



# Merge row one-hot encodings where subject-id, admittime and diagnoses_category is similar
# than count the amount of lab value labels and select the highest
print('Merging row one-hot encodings where subject-id, admittime and diagnoses_category is similar...')
lab_category_columns = [col for col in df_encoded.columns if col.startswith("lab_category_")]

def most_frequent_except_false(col):
    counts = col.value_counts()
    counts = counts.drop(False, errors='ignore') # Remove False from counts
    return counts.idxmax() if not counts.empty else False # If counts are empty (only False existed or all were empty), return False
df_majority_counts = df_encoded.groupby(['subject_id', 'admittime', 'diagnoses_category'])[lab_category_columns].apply(lambda x: x.apply(most_frequent_except_false))
df_majority_counts_flat = df_majority_counts.reset_index()
df_majority_counts_flat = df_majority_counts_flat.ffill()



# One-hot encode diagnoses
print('One-hot encoding diagnoses...')
columns_to_encode = ['diagnoses_category']
df_cleaned = pd.get_dummies(df_majority_counts_flat, columns=columns_to_encode)



# Encoding the dataset into integer class labels
# Encode "fasle, low, normal, high" → 0, 1, 2, 3 for lab tests
# Encode "false, true" → 0, 1 for diagnoses
print('Encoding the dataset into integer class labels...')
df_labeled = df_cleaned.drop(columns=['subject_id', 'admittime'])

mapping_lnh = {False: 66 ,'low': -1, 'normal': 0, 'high': 1}
low_normal_high_cols =  [col for col in df_cleaned.columns if col.startswith("lab_category_")]
df_labeled[low_normal_high_cols] = df_labeled[low_normal_high_cols].apply(lambda col: col.map(mapping_lnh).fillna(66).astype("Int64"))

mapping_tf = {False: 0, True: 1}
true_false_cols = [col for col in df_cleaned.columns if col.startswith("diagnoses_category_")]
df_labeled[true_false_cols] = df_labeled[true_false_cols].apply(lambda col: col.map(mapping_tf).fillna(66).astype("Int64"))

# All nan synonyms are replaced with nan
df_labeled.replace(66, np.nan, inplace=True)



# Fit some missing values with random forest imputer
print('Fitting missing values with random forest imputer...')
mf = MissForest(categorical=low_normal_high_cols)
mf.fit(x=df_labeled)
df_imputed = mf.transform(x=df_labeled)
df_imputed = pd.DataFrame(df_imputed, columns=df_labeled.columns)
changed_rows = (df_labeled != df_imputed).any(axis=1)
print(f"Number of imputed values in {df_imputed.isna().sum().sum() - df_labeled.isna().sum().sum()} rows")
 


# Export final dataset
print('Exporting final dataset...')
df_imputed.to_csv('data/df_train.csv', index=False)