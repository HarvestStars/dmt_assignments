import pandas as pd
import numpy as np

def process_price_feature_binned(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    """
    对 C 类字段中的 price_usd 进行区间分箱和标签化。
    默认删除原始 price_usd 列（可配置）。
    """
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    # 定义分箱边界与标签
    price_bins = [-1, 60, 80, 100, 140, 200, 300, np.inf]
    price_labels = [
        "very_cheap", "cheap", "sweet_spot", "mid_price",
        "expensive", "very_expensive", "luxury"
    ]
    price_labels_int = [
        0, 1, 2, 3, 4, 5, 6
    ]

    # 价格区间映射
    df_out["price_level"] = pd.cut(
        df_out["price_usd"],
        bins=price_bins,
        labels=price_labels_int,
        right=False,  # 区间左闭右开
        include_lowest=True
    )

    if drop_raw_columns:
        df_out.drop(columns=["price_usd"], inplace=True)

    final_columns = [
        "price_level", "promotion_flag"
    ]

    final_class_labels = [
        "price_level"
    ]

    return df_out, final_columns, final_class_labels

def process_price_feature_smoothed(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()


    final_columns = [
        "price_usd", "promotion_flag"
    ]

    final_class_labels = [
        "promotion_flag"
    ]

    return df_out, final_columns, final_class_labels

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH, nrows=100_000)
    df_c_clean = process_price_feature_binned(df, drop_raw_columns=True)
    df_c_clean.to_csv(OUT_DIR / "processed_C.csv", index=False)

    print("✅ Processed C class features and saved to split_outputs/processed_C.csv")