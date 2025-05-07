import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
SAVE_DIR = "eda_output/G_competitors"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 构造字段名 ---
comp_columns = []
for i in range(1, 9):
    comp_columns += [
        f"comp{i}_rate",
        f"comp{i}_inv",
        f"comp{i}_rate_percent_diff"
    ]
comp_columns += ["booking_bool", "click_bool"]

# --- 读取数据（仅包含这些列）---
print("Reading data...")
df = pd.read_csv(CSV_PATH, usecols=comp_columns, nrows=5_000_000)

# --- 缺失统计 ---
print("Calculating missing ratio...")
missing = df.isnull().mean().sort_values(ascending=False)
missing.to_csv(os.path.join(SAVE_DIR, "missing_ratio.csv"))

# --- 统计 Expedia 更便宜/更贵/相同 的分布（按平台聚合）---
rate_summary = {}
for i in range(1, 9):
    col = f"comp{i}_rate"
    value_counts = df[col].value_counts(normalize=True, dropna=True)
    rate_summary[col] = value_counts
rate_summary_df = pd.DataFrame(rate_summary).T.fillna(0)
rate_summary_df.to_csv(os.path.join(SAVE_DIR, "comp_rate_summary.csv"))

# --- 画出 Expedia 价格比较比例图 ---
plt.figure(figsize=(10, 5))
rate_summary_df[[1, 0, -1]].plot(kind="bar", stacked=True)
plt.title("Expedia Cheaper (=1), Same (=0), More Expensive (=-1) Proportions")
plt.ylabel("Proportion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "expedia_vs_competitors_rate_stackbar.png"))
plt.close()

# --- 关联 booking rate：当 Expedia 更便宜时是否更易被订？
print("Analyzing booking rate under comp rate conditions...")
result = {}
for i in range(1, 9):
    col = f"comp{i}_rate"
    temp = df[df[col].notnull()].groupby(col)[["booking_bool", "click_bool"]].mean()
    result[col] = temp
    temp.plot(kind="bar", title=f"{col} vs booking/click rate", figsize=(6, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{col}_target_relation.png"))
    plt.close()

# --- 衍生字段建议：多少平台 Expedia 更便宜（可用于后续建模）---
print("Calculating derived feature: expedia_better_count")
rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
df["expedia_better_count"] = df[rate_cols].apply(lambda row: (row == 1).sum(), axis=1)
df["expedia_better_count"].value_counts().sort_index().plot(kind="bar", figsize=(8, 4))
plt.title("Number of Competitors Expedia is Cheaper Than")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "expedia_better_count_dist.png"))
plt.close()

# --- 衍生字段 vs target ---
grp = df.groupby("expedia_better_count")[["booking_bool", "click_bool"]].mean()
grp.plot(kind="bar", title="expedia_better_count vs booking/click rate", figsize=(10, 5))
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "expedia_better_count_target_relation.png"))
plt.close()

print("EDA G类完成。图表输出至:", SAVE_DIR)
