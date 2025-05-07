import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/H_target"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 字段列表 ---
target_columns = [
    "booking_bool", "click_bool", "gross_bookings_usd", "position"
]

# --- 读取数据 ---
print("Reading data...")
df = pd.read_csv(CSV_PATH, usecols=target_columns, nrows=1_000_000)

# --- 缺失率 ---
df.isnull().mean().to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- booking/click 分布 ---
print("Plotting booking/click distribution...")
plt.figure(figsize=(6, 4))
df[["click_bool", "booking_bool"]].mean().plot(kind="bar")
plt.title("Click / Booking Positive Rate")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "click_booking_rate.png"))
plt.close()

# --- click vs booking 关系分析 ---
print("Analyzing inclusion relationship...")
both = df.groupby(["click_bool", "booking_bool"]).size().unstack().fillna(0)
both.to_csv(os.path.join(SAVE_DIR, "click_vs_booking_matrix.csv"))
both.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Click vs Booking Joint Distribution")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "click_vs_booking_matrix.png"))
plt.close()

# --- gross_bookings_usd 分布 ---
print("Plotting gross_bookings_usd distribution...")
plt.figure(figsize=(10, 4))
sns.histplot(df["gross_bookings_usd"].dropna(), bins=100)
plt.title("Gross Booking Value Distribution")
plt.xlim(0, np.percentile(df["gross_bookings_usd"].dropna(), 99))
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "gross_bookings_usd_dist.png"))
plt.close()

# --- gross_bookings_usd 与 booking 的关系 ---
print("Booking vs gross_bookings_usd")
mean_booking_value = df[df["booking_bool"] == 1]["gross_bookings_usd"].mean()
with open(os.path.join(SAVE_DIR, "avg_booking_value.txt"), "w") as f:
    f.write(f"Mean gross_bookings_usd for bookings: {mean_booking_value:.2f}\n")

# --- position 相关性（已在 F类中分析，这里补充）---
if "position" in df.columns:
    grp = df.groupby("position")[["booking_bool", "click_bool"]].mean()
    grp.plot(title="Position vs Target Rate", figsize=(10, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "position_vs_target.png"))
    plt.close()

print("EDA H类完成。图表输出至:", SAVE_DIR)
