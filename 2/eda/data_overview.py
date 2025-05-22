import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv_in_chunks(file_path='../dmt-2025-2nd-assignment/training_set_VU_DM.csv', chunk_size=1000_000):
    return pd.read_csv(file_path, chunksize=chunk_size)

if __name__ == "__main__":
    # 读取前 500 万条样本
    reader = read_csv_in_chunks()
    data_sample = pd.concat([chunk for i, chunk in zip(range(5), reader)])

    # 缺失率计算
    missing = data_sample.isnull().mean().sort_values(ascending=False)
    print("Missing values percentage:")
    print(missing[missing > 0])

    # -------- 可视化部分 --------

    # 1. 星级分布
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data_sample, x='prop_starrating')
    plt.title("Distribution of Hotel Star Ratings")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("eda_output/eda_starrating_dist.png")

    # 2. 点击与未点击酒店的价格分布对比
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=data_sample[data_sample['click_bool'] == 1], x='price_usd', label='Clicked', fill=True)
    sns.kdeplot(data=data_sample[data_sample['click_bool'] == 0], x='price_usd', label='Not Clicked', fill=True)
    plt.title("Price Distribution: Clicked vs. Not Clicked")
    plt.xlabel("Price (USD)")
    plt.legend()
    plt.xlim(0, 500)  # 去除极端值影响
    plt.tight_layout()
    plt.savefig("eda_output/eda_price_clicked_kde.png")

    # 3. 酒店位置评分与点击关系
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data_sample, x='click_bool', y='prop_location_score1')
    plt.title("Location Score1 vs Clicked")
    plt.xlabel("Clicked")
    plt.ylabel("Location Score1")
    plt.tight_layout()
    plt.savefig("eda_output/eda_locscore1_clicked.png")

    # 4. 缺失值热图（前20列）
    plt.figure(figsize=(12, 6))
    sns.heatmap(data_sample.iloc[:, :20].isnull(), cbar=False)
    plt.title("Missing Value Heatmap (first 20 columns)")
    plt.tight_layout()
    plt.savefig("eda_output/eda_missing_heatmap.png")

    # 5. 不同入住天数的平均预订率
    stay_booking = data_sample.groupby('srch_length_of_stay')['booking_bool'].mean()
    plt.figure(figsize=(8, 4))
    stay_booking.plot(marker='o')
    plt.title("Average Booking Rate vs Length of Stay")
    plt.xlabel("Search Length of Stay (nights)")
    plt.ylabel("Avg Booking Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda_output/eda_booking_vs_lengthofstay.png")
