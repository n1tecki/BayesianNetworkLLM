import numpy as np
from typing import Tuple
import pandas as pd


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