import pandas as pd

import pandas as pd
import numpy as np

def process_hotel_features(df: pd.DataFrame, drop_raw_columns: bool = True) -> pd.DataFrame:
    """
    对酒店字段（B类）进行缺失处理，包括构造 isnull 标志变量和分类标签。
    默认在构造后删除原始字段（可通过参数控制）。
    """
    df_processed = df.copy()

    # ---------- 1. prop_review_score 分类映射 ----------
    def map_review_score(x):
        if pd.isna(x):
            return "missing"
        elif x == 0.0:
            return "no_review"
        elif x <= 2.0:
            return "low"
        elif x <= 3.5:
            return "medium"
        elif x <= 4.5:
            return "high"
        else:
            return "very_high"

    df_processed["review_score_label"] = df_processed["prop_review_score"].apply(map_review_score)

    # ---------- 2. prop_location_score2 缺失标志 ----------
    # 删除 score1
    df_processed.drop(columns=["prop_location_score1"], inplace=True)
    # 缺失标志
    # df_processed["location_score2_missing"] = df_processed["prop_location_score2"].isnull().astype(int)
    # 填补 score2 为极小值（例如 -1）
    df_processed["location_score2_filled"] = df_processed["prop_location_score2"].fillna(-1)

    # ---------- 3. srch_query_affinity_score 缺失标志 ----------
    df_processed["query_affinity_missing"] = df_processed["srch_query_affinity_score"].isnull().astype(int)
    # df_processed["srch_query_affinity_score"] = df_processed["srch_query_affinity_score"].fillna(-999) # 值很小意味着没有曝光

    # ---------- 4. 删除原始字段（可选） ----------
    if drop_raw_columns:
        cols_to_drop = ["prop_review_score", "prop_location_score2", "srch_query_affinity_score"]
        existing = [col for col in cols_to_drop if col in df_processed.columns]
        df_processed.drop(columns=existing, inplace=True)

    return df_processed

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

