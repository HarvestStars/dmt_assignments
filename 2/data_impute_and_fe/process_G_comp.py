import pandas as pd
import numpy as np

### 合并每个propid下的所有 compX 信息，也就是我们只查看这个酒店在市场其他平台的报价，库存情况，来决定Expedia平台是否占优

# 举例:
# 胜出次数（compX_rate == +1）
# 平价次数（compX_rate == 0）
# 劣势次数（compX_rate == -1）
# 有效平台数量（非空）

# 然后计算winrate：
# winrate = 胜出平台数 / 有效平台数
# loserate = 劣势平台数 / 有效平台数

def add_competition_features(df: pd.DataFrame, drop_raw_comp_columns: bool = True, non_copy: bool = True) -> pd.DataFrame:
    # 构造字段名
    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    comp_percent_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

    def compute_competition(row):
        rate_values = row[comp_rate_cols].dropna().values # null signifies there is no competitive data
        inv_values = row[comp_inv_cols].dropna().values

        total_rate = len(rate_values)
        win = np.sum(rate_values == 1)
        lose = np.sum(rate_values == -1)

        if total_rate == 0:
            winrate = np.nan
            loserate = np.nan
            comp_label = -1
        else:
            winrate = win / total_rate
            loserate = lose / total_rate
            if winrate > 0.8:
                comp_label = 3
            elif winrate > 0.5:
                comp_label = 2
            elif loserate > 0.5:
                comp_label = 0
            else:
                comp_label = 1

        total_inv = len(inv_values)
        advantage = np.sum(inv_values == 1)
        common = np.sum(inv_values == 0)
        inventory_winrate = advantage / total_inv if total_inv > 0 else np.nan # null signifies there is no competitive data

        return pd.Series({
            "comp_label": comp_label,
            "comp_winrate": winrate,
            "comp_inventory_winrate": inventory_winrate
        })

    # 应用函数
    comp_features = df.apply(compute_competition, axis=1)
    df_result = pd.concat([df, comp_features], axis=1)
   
    # 添加 no_comp 标志 
    # 0表示有竞争数据，1表示没有竞争数据
    # 1可能是极强的购买信号?
    df_result["no_comp"] = (
        df_result["comp_winrate"].isnull() & df_result["comp_inventory_winrate"].isnull()
    ).astype(int)

    # 根据开关删除原始字段
    if drop_raw_comp_columns:
        cols_to_drop = comp_rate_cols + comp_inv_cols + comp_percent_cols
        existing = [col for col in cols_to_drop if col in df_result.columns]
        df_result.drop(columns=existing, inplace=True)

    final_columns = [
        "comp_label",
        "comp_winrate",
        "comp_inventory_winrate",
        "no_comp"
    ]
    final_class_labels = [
        "comp_label",
        "comp_winrate",
        "comp_inventory_winrate",
        "no_comp"
    ]

    return df_result, final_columns, final_class_labels


if __name__ == "__main__":
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt

    # --- 主流程：导入原始数据 + 应用 ---
    CSV_PATH = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"
    OUT_DIR = Path("split_outputs")
    OUT_DIR.mkdir(exist_ok=True)

    chunksize = 1_000_000
    # reader = pd.read_csv(CSV_PATH, chunksize=chunksize)
    raw_df = pd.read_csv(CSV_PATH, nrows=chunksize)


    # for i, chunk in enumerate(reader):
    # print(f"🔄 Processing chunk {i} (rows {i * chunksize:,} ~ {(i+1) * chunksize - 1:,})")

    # 添加竞争特征
    df, _ = add_competition_features(raw_df, drop_raw_comp_columns=True)
    print(f"✅ Processed chunk {chunksize} lines with competition features.")
    
    # 导出带特征的分块文件
    # output_path = OUT_DIR / f"train_with_comp_features_part_{i}.csv"
    # df_with_comp.to_csv(output_path, index=False)
    # print(f"✅ Saved to {output_path}")


    # --- 可选：分析竞争特征与目标变量的关系 ---
    # 1. comp_label vs booking/click rate
    print("📊 comp_label vs booking/click rate")
    label_stats = df.groupby("comp_label")[["booking_bool", "click_bool"]].mean()
    label_stats.plot(kind="bar", title="comp_label vs booking/click rate", figsize=(6, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comp_label_target_relation.png"))
    plt.close()

    # 2. comp_winrate 分箱后分析
    print("📊 comp_winrate bins vs booking/click rate")
    df["comp_winrate_bin"] = pd.cut(df["comp_winrate"], bins=[-0.01, 0.2, 0.5, 0.8, 1.0], labels=["low", "mid", "high", "dominant"])
    winrate_stats = df.groupby("comp_winrate_bin")[["booking_bool", "click_bool"]].mean()
    winrate_stats.plot(kind="bar", title="comp_winrate_bin vs booking/click rate", figsize=(6, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comp_winrate_target_relation.png"))
    plt.close()

    # 3. comp_inventory_winrate 分箱后分析
    print("📊 comp_inventory_winrate bins vs booking/click rate")
    df["comp_inv_winrate_bin"] = pd.cut(df["comp_inventory_winrate"], bins=[-0.01, 0.2, 0.5, 0.8, 1.0], labels=["low", "mid", "high", "dominant"])
    inv_stats = df.groupby("comp_inv_winrate_bin")[["booking_bool", "click_bool"]].mean()
    inv_stats.plot(kind="bar", title="comp_inventory_winrate_bin vs booking/click rate", figsize=(6, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comp_inv_winrate_target_relation.png"))
    plt.close()

    # 4. no_comp 标志行为差异
    print("📊 no_comp (0/1) vs booking/click rate")
    no_comp_stats = df.groupby("no_comp")[["booking_bool", "click_bool"]].mean()
    no_comp_stats.plot(kind="bar", title="no_comp vs booking/click rate", figsize=(6, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "no_comp_target_relation.png"))
    plt.close()

    print("✅ All analysis plots saved.")
