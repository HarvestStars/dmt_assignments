import pandas as pd

import pandas as pd
import numpy as np

def process_hotel_features(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    """
    对酒店字段（B类）进行缺失处理，包括构造 isnull 标志变量和分类标签。
    默认在构造后删除原始字段（可通过参数控制）。
    """
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    # ---------- 1. prop_review_score 分类映射 ----------
    def map_review_score(x):
        if pd.isna(x):
            return 0 # "missing"
        elif x == 0.0:
            return 1 # "no_review"
        elif x <= 2.0:
            return 2 # "low"
        elif x <3.5:    # [2.0 - 3.5) 中分段
            return 3 # "medium"
        elif x <= 4.5:  # [3.5 - 4.5] 高分段
            return 4 # "high"
        else:
            return 5 # "very_high"

    df_out["review_score_label"] = df_out["prop_review_score"].apply(map_review_score)

    # ---------- 2. prop_location_score2 缺失标志 ----------
    # 删除 score1
    df_out.drop(columns=["prop_location_score1"], inplace=True)
    # 缺失标志
    # df_processed["location_score2_missing"] = df_processed["prop_location_score2"].isnull().astype(int)
    # 填补 score2 为极小值（例如 -1）
    df_out["location_score2_filled"] = df_out["prop_location_score2"].fillna(-1)

    # ---------- 3. srch_query_affinity_score 缺失标志 ----------
    df_out["query_affinity_missing"] = df_out["srch_query_affinity_score"].isnull().astype(int)
    # df_processed["srch_query_affinity_score"] = df_processed["srch_query_affinity_score"].fillna(-999) # 值很小意味着没有曝光

    # ---------- 4. prop_log_historical_price 分类映射 ----------
    def map_price_level(x):
        if x == 0:
            return 0 # "zero"
        elif x < 3:
            return 1 # "very_low"
        elif x < 4.5:
            return 2 # "low"
        elif x < 5.5:
            return 3 # "mid"
        else:
            return 4 # "high"
    df_out["historical_price_level"] = df_out["prop_log_historical_price"].apply(map_price_level)

    # ---------- 5. 删除原始字段（可选） ----------
    if drop_raw_columns:
        cols_to_drop = ["prop_review_score", "prop_location_score2", "srch_query_affinity_score", "prop_log_historical_price"]
        existing = [col for col in cols_to_drop if col in df_out.columns]
        df_out.drop(columns=existing, inplace=True)

    final_cols = [
        "prop_id", "review_score_label","location_score2_filled", "historical_price_level", "query_affinity_missing"
    ]
    return df_out, final_cols

if __name__ == "__main__":
    import os
    from pathlib import Path
    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH, nrows=100_000)
    df_b_clean = process_hotel_features(df, drop_raw_columns=True)
    df_b_clean.to_csv(OUT_DIR / "processed_B.csv", index=False)
    print("Processed B class features and saved to split_outputs/processed_B.csv")

