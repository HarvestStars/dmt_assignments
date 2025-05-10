import pandas as pd

### 1 直接用中位值填充缺失（分布数据的中位值）
### 2 聚类或分类用户 → 用邻近类中有值者均值填充（相同的srch等其他类字段代表了相同的用户，所以直接赋值）KMeans 建议放到后期优化中尝试
### 3 仅构造 is_new_user 标志 + 保留原字段为空 最快的baseline方案，本次采用
def process_new_user(df: pd.DataFrame, drop_raw_columns: bool = False, non_copy: bool = True) -> pd.DataFrame:
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()
    df_out.drop(columns=["visitor_location_country_id"], inplace=True)

    # 判断是否新用户（两列都缺失）
    df_out["is_new_user"] = (
        df_out["visitor_hist_starrating"].isnull() &
        df_out["visitor_hist_adr_usd"].isnull()
    ).astype(int)

    # 填充缺失值
    df_out["visitor_hist_starrating"].fillna(0, inplace=True)
    df_out["visitor_hist_adr_usd"].fillna(0, inplace=True)

    final_columns = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "is_new_user"
    ]
    final_class_labels = [
        "is_new_user"
    ]

    return df_out, final_columns, final_class_labels

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH, nrows=100_000)
    df_d_clean = process_new_user(df, drop_raw_columns=False)
    df_d_clean.to_csv(OUT_DIR / "processed_D.csv", index=False)

    print("✅ Processed D class (user info) and saved to split_outputs/processed_D.csv")
