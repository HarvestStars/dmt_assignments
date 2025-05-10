import pandas as pd
import numpy as np

def process_random_feature(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    final_columns = [
        "random_bool"
    ]

    if drop_raw_columns:
        df_out.drop(columns=["random_bool"], inplace=True)
        final_columns.remove("random_bool")

    return df_out, final_columns, []
