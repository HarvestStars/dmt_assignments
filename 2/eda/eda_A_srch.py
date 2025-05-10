import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/A_srch"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- A类字段列表 ---
search_columns = [
    "srch_id", "date_time", "site_id", "visitor_location_country_id",
    "srch_destination_id", "srch_length_of_stay", "srch_booking_window",
    "srch_adults_count", "srch_children_count", "srch_room_count",
    "srch_saturday_night_bool", "srch_query_affinity_score",
    "booking_bool", "click_bool"
]

# --- 读取前500万行 ---
print("Reading data...")
reader = pd.read_csv(CSV_PATH, usecols=search_columns, nrows=5_000_000, parse_dates=["date_time"])
df = reader.copy()

# --- 缺失值统计 ---
print("Calculating missing value ratio...")
missing = df.isnull().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- 时间特征衍生 ---
print("Extracting time features...")
df["search_month"] = df["date_time"].dt.month
df["search_hour"] = df["date_time"].dt.hour
df["search_weekday"] = df["date_time"].dt.dayofweek  # 0=Monday

# --- 数值型分布可视化 ---
print("Plotting numeric distributions...")
num_cols = [
    "srch_length_of_stay", "srch_booking_window",
    "srch_adults_count", "srch_children_count", "srch_room_count",
    "srch_query_affinity_score", "search_hour", "search_month", "search_weekday"
]

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=50, kde=True)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_dist.png"))
    plt.close()

# --- 类别型分布可视化 ---
print("Plotting categorical distributions...")
cat_cols = ["site_id", "visitor_location_country_id", "srch_saturday_night_bool"]

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    df[col].value_counts(normalize=True).head(20).plot(kind="bar")
    plt.title(f"{col} Frequency (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_freq.png"))
    plt.close()

# --- 分组 booking/click 概率分析（带分箱） ---
print("Plotting booking/click groupby average...")
for col in num_cols + cat_cols:
    plot_col = col  # 默认用原始列名
    max_val = df[col].max()
    step = 5  # 每5天一个段
    bin_edges = list(range(0, int(max_val) + step, step))
    bin_col = f"{col}_binned"
    df[bin_col] = pd.cut(df[col], bins=bin_edges, right=False)
    # df[bin_col] = pd.cut(df[col], bins=30)  # 分30个箱
    plot_col = bin_col

    grp = df.groupby(plot_col)[["booking_bool", "click_bool"]].mean()
    grp.plot(kind="bar", figsize=(12, 5), title=f"{col} (binned) vs booking/click rate")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_target_relation.png"))
    plt.close()

print("EDA A类完成。图表和表格输出至:", SAVE_DIR)
