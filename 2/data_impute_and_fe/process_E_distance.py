import pandas as pd
import numpy as np

def process_distance_feature(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    if drop_raw_columns:
        df_out.drop(columns=["orig_destination_distance"], inplace=True)

    return df_out, []
