import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/F_sort"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 字段列表 ---
sort_columns = ["position", "random_bool", "booking_bool", "click_bool"]

# --- 数据读取 ---
print("Reading data...")
df = pd.read_csv(CSV_PATH, usecols=sort_columns, nrows=5_000_000)

# --- 缺失值检查 ---
df.isnull().mean().to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- Position 分布 ---
print("Plotting position distribution...")
plt.figure(figsize=(10, 5))
sns.histplot(df["position"], bins=50)
plt.title("Hotel Position Distribution")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "position_dist.png"))
plt.close()

# --- Position vs click/booking ---
print("Plotting position vs booking/click...")
grp = df.groupby("position")[["booking_bool", "click_bool"]].mean()
grp.plot(figsize=(10, 5), title="Position vs Booking/Click Rate")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "position_target_relation.png"))
plt.close()

# --- random_bool 分布 ---
plt.figure(figsize=(6, 4))
df["random_bool"].value_counts(normalize=True).plot(kind="bar")
plt.title("Random Bool Frequency")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "random_bool_freq.png"))
plt.close()

# --- random_bool vs click/booking ---
print("Plotting random_bool impact...")
grp2 = df.groupby("random_bool")[["booking_bool", "click_bool"]].mean()
grp2.plot(kind="bar", figsize=(6, 4), title="Random Bool vs Booking/Click Rate")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "random_bool_target_relation.png"))
plt.close()

print("EDA F类完成。图表输出至:", SAVE_DIR)
