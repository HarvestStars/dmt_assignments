import pandas as pd

def read_csv_in_chunks(file_path='./test_competition_features.csv', chunk_size=1000_000):
    return pd.read_csv(file_path, chunksize=chunk_size)

if __name__ == "__main__":
    reader = read_csv_in_chunks()
    data_sample = pd.concat([chunk for i, chunk in zip(range(5), reader)])  # 取前100万条样本
    missing = data_sample.isnull().mean().sort_values(ascending=False)
    print("Missing values percentage:")
    print(missing[missing > 0])
