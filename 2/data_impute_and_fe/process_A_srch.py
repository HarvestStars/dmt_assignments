import pandas as pd

def process_search_behavior_features(df: pd.DataFrame, drop_raw_columns: bool = True) -> pd.DataFrame:
    df_out = df.copy()

    # -- 分箱1: 预计停留天数 (srch_length_of_stay) --
    stay_bins = [0, 1, 3, 5, 7, 10, 15, 30, 45, 60]
    stay_labels = [
        "1_day", "2-3", "4-5", "6-7", "8-10",
        "11-15", "16-30", "31-45", "46_plus"
    ]
    df_out["stay_length_label"] = pd.cut(
        df_out["srch_length_of_stay"],
        bins=stay_bins,
        labels=stay_labels,
        right=False,
        include_lowest=True
    )

    # -- 分箱2: 提前预定天数 (srch_booking_window) --
    booking_bins = [-1, 0, 3, 7, 14, 30, 90, 180, 420, 460, 500]
    booking_labels = [
        "same_day", "1-3", "4-7", "8-14", "15-30",
        "31-90", "91-180", "181-420", "421-460", "460_plus"
    ]
    df_out["booking_window_label"] = pd.cut(
        df_out["srch_booking_window"],
        bins=booking_bins,
        labels=booking_labels,
        right=False,
        include_lowest=True
    )

    # 删除原始列（可选）
    if drop_raw_columns:
        df_out.drop(columns=["srch_length_of_stay", "srch_booking_window"], inplace=True)


    return df_out

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH, nrows=100_000)
    df_a_clean = process_search_behavior_features(df, drop_raw_columns=True)
    df_a_clean.to_csv(OUT_DIR / "processed_A.csv", index=False)

    print("✅ Processed A class features and saved to split_outputs/processed_A.csv")