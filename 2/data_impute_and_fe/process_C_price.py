import pandas as pd
import numpy as np

def process_price_feature(df: pd.DataFrame, drop_raw_columns: bool = True) -> pd.DataFrame:
    """
    对 C 类字段中的 price_usd 进行区间分箱和标签化。
    默认删除原始 price_usd 列（可配置）。
    """
    df_out = df.copy()

    # 定义分箱边界与标签
    price_bins = [-1, 60, 80, 100, 140, 200, 300, np.inf]
    price_labels = [
        "very_cheap", "cheap", "sweet_spot", "mid_price",
        "expensive", "very_expensive", "luxury"
    ]

    # 价格区间映射
    df_out["price_level"] = pd.cut(
        df_out["price_usd"],
        bins=price_bins,
        labels=price_labels,
        right=False,  # 区间左闭右开
        include_lowest=True
    )

    if drop_raw_columns:
        df_out.drop(columns=["price_usd"], inplace=True)

    return df_out

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH, nrows=100_000)
    df_c_clean = process_price_feature(df, drop_raw_columns=True)
    df_c_clean.to_csv(OUT_DIR / "processed_C.csv", index=False)

    print("✅ Processed C class features and saved to split_outputs/processed_C.csv")