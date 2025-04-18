import pandas as pd

def imputing_with_zero(df_cleaned):
    df_method1 = df_cleaned.copy() # keep it clean
    return df_method1.fillna(0) # 方法 1：将所有缺失值填充为 0

def imputing_with_mean(df_cleaned: pd.DataFrame):
    """
    Impute missing values in the dataset with the mean of each column.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame with missing values.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values imputed.
    """
    # 方法 2：将所有值为 0 的位置也视为缺失，先替换成 NaN，再按用户取均值填充
    df_method2 = df_cleaned.copy()

    # 定义待处理的特征列（去除 id, date, mood）
    feature_cols = df_method2.columns.difference(['id', 'date', 'mood'])

    # 将所有 0 替换为 NaN（假设 0 为缺失）
    df_method2[feature_cols] = df_method2[feature_cols].replace(0, pd.NA)

    # 按用户 id 分组，对每列用组内均值填充
    df_method2[feature_cols] = df_method2.groupby('id')[feature_cols].transform(lambda x: x.fillna(x.mean()))
    
    # 兜底操作：如果还有 NaN（比如某个 id 的某列全部为 NaN），用 0 填充
    df_method2[feature_cols] = df_method2[feature_cols].fillna(0)

    return df_method2 # 返回填充后的 DataFrame

def imputing_with_removal_NaN(df_cleaned):
    """
    Remove rows with NaN values in the dataset.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame with missing values.
    
    Returns:
    pd.DataFrame: The DataFrame with rows containing NaN values removed.
    """
    # 方法 3：去除除了 mood 外其他特征全为 0 且 activity 是缺失值的记录
    df_method3 = df_cleaned.copy()

    # 定义特征列（去除 id, date, mood）
    feature_cols = df_method3.columns.difference(['id', 'date', 'mood'])

    # 创建布尔条件：其他特征全部为 0 且 activity 为空
    condition = (df_method3[feature_cols].sum(axis=1) == 0) & (df_method3['activity'].isna())

    # 过滤掉这些“无用记录”
    df_method3 = df_method3[~condition].reset_index(drop=True)
    return df_method3 # 返回去除后的 DataFrame

if __name__ == "__main__":
    import data_load
    import processing

    # 加载数据
    df = data_load.load_mood_dataset("../../raw_data/dataset_mood_smartphone.csv")
    
    # 查看结构
    print(df.head(3))

    # 快速了解缺失情况
    print("Original data:", df.isnull().mean().sort_values(ascending=False))

    # # 随机选择一个变量列并统计非空值数量
    # column_name, non_null_count = inspect_random_variable_column(df, column_name="appCat.entertainment")
    # print(f"随机选中的列: {column_name}, 非空值数量: {non_null_count}")

    # 删除缺失率高于 90% 的列
    cleaned_df = processing.drop_high_nan_columns(df, threshold=0.995)

    # 保存 删除缺失 后的数据
    cleaned_df.to_csv('../../raw_data/cleaned_data.csv', index=False)

    # 并按天聚合的数据
    df_daily = processing.aggregate_by_day(cleaned_df)
    df_daily.to_csv("../../raw_data/cleaned_data_daily_summary.csv", index=False)
    print("df_daily data:", df_daily.isnull().mean().sort_values(ascending=False))        # 快速了解缺失情况

    # 聚合后，二次清洗mood为空的数据
    print(f"原始按天聚合数据共 {len(df_daily)} 条")
    df_daily = df_daily[df_daily['mood'].notna()].reset_index(drop=True)
    print(f"保留含有 mood 值的数据共 {len(df_daily)} 条")
    df_daily.to_csv("../../raw_data/cleaned_data_daily_summary_mood.csv", index=False)

    # 二次清理后，还剩下一些缺失情况
    print("df_daily data:", df_daily.isnull().mean().sort_values(ascending=False))        # 快速了解缺失情况

    # 进行缺失值填充
    df_method1 = imputing_with_zero(df_daily)
    df_method2 = imputing_with_mean(df_daily)
    df_method3 = imputing_with_removal_NaN(df_daily)
    df_method2.to_csv("../../raw_data/cleaned_data_daily_summary_mood_imputed.csv", index=False)
    
    print("Method 1 shape:", df_method1.shape)
    print("Method 2 shape:", df_method2.shape)
    print("Method 3 shape:", df_method3.shape)
