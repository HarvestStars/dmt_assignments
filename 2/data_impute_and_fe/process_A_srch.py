import pandas as pd
import numpy as np

def process_search_behavior_features_binned(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    # -- 分箱1: 预计停留天数 (srch_length_of_stay) --
    stay_bins = [0, 5, 10, 15, 30, 60]
    stay_labels = [
        "1-5", "5-10", "10-15",
        "15-30", "30_plus"
    ]
    stay_labels_int = [
        1, 2, 3,
        4, 5
    ]

    df_out["stay_length_label"] = pd.cut(
        df_out["srch_length_of_stay"],
        bins=stay_bins,
        labels=stay_labels_int,
        right=False,
        include_lowest=True
    )

    # -- 分箱2: 提前预定天数 (srch_booking_window) --
    booking_bins = [0, 5, 10, 15, 360, 375, 380, 400, 460, 466, 500]
    booking_labels = [
        "0-5", "5-10", "10-15",
        "15-360", "360-375", "35-380",
        "380-400", "400-460", "460-466",
        "466_500"
    ]
    booking_labels_int = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10
    ]
    df_out["booking_window_label"] = pd.cut(
        df_out["srch_booking_window"],
        bins=booking_bins,
        labels=booking_labels_int,
        right=False,
        include_lowest=True
    )

    # 删除原始列（可选）
    if drop_raw_columns:
        df_out.drop(columns=["srch_length_of_stay", "srch_booking_window"], inplace=True)

    final_columns = [
        "srch_id", 
        "stay_length_label", "booking_window_label"
    ]

    final_class_labels = [
        "stay_length_label", "booking_window_label"
    ]

    return df_out, final_columns, final_class_labels

def process_search_features_smoothed(df: pd.DataFrame, drop_raw_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    df_out = None
    if non_copy:
        df_out = df
    else:
        df_out = df.copy()

    # drop raw columns
    df_out.drop(columns=["date_time", "site_id", "srch_room_count"], inplace=True)

    # add smoothed features
    df_out["total_guests"] = df_out["srch_adults_count"] + df_out["srch_children_count"]
    df_out.drop(columns=["srch_adults_count", "srch_children_count"], inplace=True)

    final_columns = [
        "srch_id", "srch_destination_id", "srch_length_of_stay",
        "srch_booking_window", "total_guests",
        "srch_saturday_night_bool"
    ]

    return df_out, final_columns, []

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH, nrows=100_000)
    df_a_clean = process_search_behavior_features_binned(df, drop_raw_columns=True)
    df_a_clean.to_csv(OUT_DIR / "processed_A.csv", index=False)

    print("✅ Processed A class features and saved to split_outputs/processed_A.csv")