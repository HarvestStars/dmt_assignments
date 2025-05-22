import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_csv_in_chunks(file_path='../dmt-2025-2nd-assignment/training_set_VU_DM.csv', chunk_size=1000_000):
    return pd.read_csv(file_path, chunksize=chunk_size)

if __name__ == "__main__":
    # 读取前 500 万条样本
    reader = read_csv_in_chunks()
    data_sample = pd.concat([chunk for i, chunk in zip(range(1), reader)])

    # 选择用于协方差分析的字段（去掉分类字段如prop_id, prop_country_id）
    prop_features = [
        "prop_starrating", "prop_review_score", "prop_brand_bool",
        "prop_location_score1", "prop_location_score2", "prop_log_historical_price",
        "price_usd"
    ]

    df = data_sample[prop_features].dropna()  # 简化处理：删除缺失值

    # --------- Pearson相关性热图 ---------
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Pearson Correlation Matrix (Hotel Attributes)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("eda_output/corr/B/eda_prop_pearson_corr.png")

    # --------- 协方差热图 ---------
    # plt.figure(figsize=(10, 8))
    # cov_matrix = df.cov()
    # sns.heatmap(cov_matrix, cmap="YlGnBu", annot=True, fmt=".1f", square=True)
    # plt.title("Covariance Matrix (Hotel Attributes)", fontsize=16)
    # plt.tight_layout()
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.savefig("eda_output/corr/B/eda_prop_cov_matrix.png")

    # --------- 成对字段分布关系图 ---------
    subset = df[["prop_starrating", "prop_review_score", "prop_location_score1", "prop_log_historical_price"]]
    sns.pairplot(subset, diag_kind='kde', corner=True)
    plt.suptitle("Pairwise Distribution of Key Hotel Features", y=1.02)
    plt.savefig("eda_output/corr/B/eda_prop_pairplot.png")
