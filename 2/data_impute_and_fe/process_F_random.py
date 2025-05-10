import pandas as pd
import numpy as np

def process_random_feature(df: pd.DataFrame, drop_raw_columns: bool = True) -> pd.DataFrame:
    df_out = df.copy()

    if drop_raw_columns:
        df_out.drop(columns=["random_bool"], inplace=True)

    return df_out
