import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/B_prop"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- B类字段列表 ---
prop_columns = [
    "prop_id", "prop_starrating", "prop_review_score", "prop_brand_bool",
    "prop_location_score1", "prop_location_score2", "prop_country_id",
    "prop_log_historical_price", "booking_bool", "click_bool"
]

# --- 读取数据 ---
print("Reading data...")
reader = pd.read_csv(CSV_PATH, usecols=prop_columns, nrows=5_000_000)
df = reader.copy()

# --- 缺失值统计 ---
print("Calculating missing value ratio...")
missing = df.isnull().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- 分布分析 ---
print("Plotting distributions...")
num_cols = [
    "prop_starrating", "prop_review_score",
    "prop_location_score1", "prop_location_score2",
    "prop_log_historical_price"
]
cat_cols = [
    "prop_brand_bool", "prop_country_id"
]

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), bins=50, kde=True)
    plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_dist.png"))
    plt.close()

for col in cat_cols:
    plt.figure(figsize=(10, 4))
    df[col].value_counts(normalize=True).head(20).plot(kind="bar")
    plt.title(f"{col} Frequency (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_freq.png"))
    plt.close()

# --- booking/click 分组分析 ---
print("Plotting feature vs target relations...")
for col in num_cols + cat_cols:
    grp = df.groupby(col)[["booking_bool", "click_bool"]].mean()
    grp.plot(kind="bar", figsize=(10, 5), title=f"{col} vs booking/click rate")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_target_relation.png"))
    plt.close()

print("EDA B类完成。图表和表格输出至:", SAVE_DIR)
