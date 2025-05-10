import pandas as pd
import numpy as np

def process_train_label(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
   
    def construct_label(row):
        if row['booking_bool'] == 1:
            return 5
        elif row['click_bool'] == 1:
            return 1
        else:
            return 0

    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    df_out['train_relevance'] = df_out.apply(construct_label, axis=1)

    # 删掉原始字段
    if drop_raw_columns:
        df_out.drop(columns=["booking_bool", "click_bool"], inplace=True)

    return df_out, ["train_relevance"]
