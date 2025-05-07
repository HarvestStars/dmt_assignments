import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/D_userhist"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- D类字段 ---
user_hist_columns = [
    "visitor_hist_starrating", "visitor_hist_adr_usd",
    "booking_bool", "click_bool"
]

# --- 读取数据 ---
print("Reading data...")
reader = pd.read_csv(CSV_PATH, usecols=user_hist_columns, nrows=5_000_000)
df = reader.copy()

# --- 缺失统计 ---
print("Calculating missing ratios...")
missing = df.isnull().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- 构造缺失指示变量 ---
print("Creating isnull indicators...")
df["star_missing"] = df["visitor_hist_starrating"].isnull().astype(int)
df["adr_missing"] = df["visitor_hist_adr_usd"].isnull().astype(int)

# --- 缺失 vs 点击/预订 比率 ---
print("Plotting null indicator relations...")
for col in ["star_missing", "adr_missing"]:
    grp = df.groupby(col)[["booking_bool", "click_bool"]].mean()
    grp.plot(kind="bar", figsize=(6, 4), title=f"{col} vs booking/click rate")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_target_relation.png"))
    plt.close()

# --- 非缺失值的分布分析 ---
print("Plotting distributions of available data...")
df_starrate = df[df["star_missing"] == 0]
df_adr = df[df["adr_missing"] == 0]

# 星级分布
plt.figure(figsize=(6, 4))
sns.histplot(df_starrate["visitor_hist_starrating"], bins=20, kde=True)
plt.title("visitor_hist_starrating Distribution")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "visitor_hist_starrating_dist.png"))
plt.close()

# 平均房价分布
plt.figure(figsize=(6, 4))
sns.histplot(df_adr["visitor_hist_adr_usd"], bins=50, kde=True)
plt.title("visitor_hist_adr_usd Distribution")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "visitor_hist_adr_usd_dist.png"))
plt.close()

# 非缺失值 vs booking/click rate
print("Plotting value-binned target relation...")
for col in ["visitor_hist_starrating", "visitor_hist_adr_usd"]:
    grp = df[df[col].notnull()].groupby(pd.qcut(df[col], 10, duplicates='drop'))[["booking_bool", "click_bool"]].mean()
    grp.plot(kind="bar", figsize=(10, 5), title=f"{col} vs booking/click rate")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_target_relation.png"))
    plt.close()

print("EDA D类完成。图表输出至:", SAVE_DIR)
