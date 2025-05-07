import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/E_distance"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 字段列表 ---
geo_columns = [
    "orig_destination_distance", "booking_bool", "click_bool"
]

# --- 数据读取 ---
print("Reading data...")
df = pd.read_csv(CSV_PATH, usecols=geo_columns, nrows=5_000_000)

# --- 缺失统计 ---
print("Calculating missing ratio...")
missing = df.isnull().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- 构造缺失指示变量 ---
df["distance_isnull"] = df["orig_destination_distance"].isnull().astype(int)

# --- 缺失 vs booking/click 率 ---
print("Plotting missing indicator impact...")
grp = df.groupby("distance_isnull")[["booking_bool", "click_bool"]].mean()
grp.plot(kind="bar", figsize=(6, 4), title="Missing Distance vs Booking/Click")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "missing_indicator_relation.png"))
plt.close()

# --- 分布图（去掉极端值） ---
print("Plotting distance distribution...")
plt.figure(figsize=(8, 4))
sns.histplot(df["orig_destination_distance"].dropna(), bins=100, kde=True)
plt.xlim(0, np.percentile(df["orig_destination_distance"].dropna(), 99))  # 去除极值
plt.title("Distance Distribution (trimmed)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "distance_dist_trimmed.png"))
plt.close()

# --- 距离分箱后观察点击/预订率 ---
print("Plotting distance vs booking/click relation...")
df_valid = df[df["orig_destination_distance"].notnull()]
df_valid["distance_bin"] = pd.qcut(df_valid["orig_destination_distance"], q=10, duplicates='drop')
grp2 = df_valid.groupby("distance_bin")[["booking_bool", "click_bool"]].mean()
grp2.plot(kind="bar", figsize=(12, 5), title="Distance Quantiles vs Booking/Click")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "distance_target_relation.png"))
plt.close()

print("EDA E类完成。图表输出至:", SAVE_DIR)
