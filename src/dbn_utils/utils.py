import numpy as np
from typing import Tuple
import pandas as pd
import os


def split_train_test(df: pd.DataFrame,
                             test_size: float = 0.3,
                             random_state: int = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # get unique hadm_id values
    unique_ids = df.index.unique().to_numpy()
    # shuffle
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_ids)
    # split point
    split_at = int(len(unique_ids) * (1 - test_size))
    train_ids = unique_ids[:split_at]
    test_ids  = unique_ids[split_at:]
    # select by index
    train_df = df.loc[train_ids]
    test_df  = df.loc[test_ids]
    return train_df, test_df


def create_train_test_set(
        df: pd.DataFrame, 
        cols: list, 
        export_path: str, 
        test_size: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_copy = df.copy()
    df_copy.index.name = 'hadm_id'

    # Reading as int8 to reduce memory usage
    cat_cols = ['sepsis'] + cols
    for c in cat_cols:
        df_copy[c] = pd.Categorical(df_copy[c]).codes.astype('int8')

    # Split into training and test data
    df_train, df_test = split_train_test(df_copy, 
                            test_size=test_size, 
                            random_state=42
                        )
    
    df_train.to_parquet(os.path.join(export_path, 'df_train.parquet'), engine='pyarrow')
    df_test.to_parquet(os.path.join(export_path, 'df_test.parquet'), engine='pyarrow')
    return df_train, df_test