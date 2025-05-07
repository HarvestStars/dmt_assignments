import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/C_price"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- C类字段列表 ---
price_columns = [
    "price_usd", "promotion_flag", "prop_log_historical_price",
    "gross_bookings_usd", "booking_bool", "click_bool"
]

# --- 数据读取 ---
print("Reading data...")
reader = pd.read_csv(CSV_PATH, usecols=price_columns, nrows=5_000_000)
df = reader.copy()

# --- 缺失统计 ---
print("Calculating missing ratio...")
missing = df.isnull().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- price/historical price/log ratio ---
print("Calculating price-related features...")
df["historical_price"] = df["prop_log_historical_price"].apply(lambda x: pd.NA if pd.isna(x) else 10 ** x)
df["price_ratio"] = df["price_usd"] / df["historical_price"]

# --- 分布图 ---
print("Plotting distributions...")
cols_to_plot = ["price_usd", "historical_price", "price_ratio", "gross_bookings_usd"]
for col in cols_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=100, kde=True)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_dist.png"))
    plt.close()

# --- promotion_flag 分布 ---
plt.figure(figsize=(6, 4))
df["promotion_flag"].value_counts(normalize=True).plot(kind="bar")
plt.title("Promotion Flag Frequency")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "promotion_flag_freq.png"))
plt.close()

# --- 各字段 vs 预订/点击 比率 ---
print("Plotting feature vs target relations...")
for col in ["price_usd", "price_ratio", "promotion_flag"]:
    grp = df.groupby(pd.qcut(df[col], 10, duplicates='drop'))[["booking_bool", "click_bool"]].mean() \
            if df[col].nunique() > 10 else df.groupby(col)[["booking_bool", "click_bool"]].mean()
    grp.plot(kind="bar", figsize=(10, 5), title=f"{col} vs booking/click rate")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_target_relation.png"))
    plt.close()

print("EDA C类完成。图表输出至:", SAVE_DIR)
