def aggregate_by_day(df):
    """
    按照 'id' 和 'date' 维度合并数据：
    - mood, arousal, valence, activity：取均值（保留 NaN）
    - 其他所有列（如使用时间、call、sms）：缺失值填0，再求和
    
    参数:
        df: DataFrame - 宽格式的变量表，含有 id, time, 各类变量列
        
    返回:
        df_daily: DataFrame - 合并后的按天聚合数据
    """
    # 深拷贝，避免污染原始数据
    df = df.copy()
    df['date'] = df['time'].dt.date  # 提取日期

    # 定义取均值的变量
    mean_vars = {'mood', 'circumplex.arousal', 'circumplex.valence', 'activity'}
    
    # 将总量类的变量 NaN → 0 (只对非 mean_vars), 意味着对于累加和型变量，我们假设缺失值等价于 "没有使用"
    for col in df.columns:
        if col not in mean_vars and col not in ['id', 'time', 'date']:
            df[col] = df[col].fillna(0)

    # 构造聚合字典
    agg_dict = {
        col: ('mean' if col in mean_vars else 'sum')
        for col in df.columns
        if col not in ['id', 'time', 'date']
    }

    # 聚合
    df_daily = df.groupby(['id', 'date']).agg(agg_dict).reset_index()
    
    return df_daily

def drop_high_nan_columns(df, threshold=0.95, exclude_columns=['mood']):
    """
    删除缺失率高于 threshold 的列（除非列在 exclude_columns 中）

    参数:
    - df: DataFrame，要处理的数据
    - threshold: 缺失率阈值（例如 0.9 表示90%）
    - exclude_columns: 不参与删除的列名列表

    返回:
    - 一个删除了高缺失列后的新 DataFrame
    """
    # 计算每一列缺失值的比例
    nan_ratio = df.isna().mean()
    
    # 找到要删除的列：缺失率高于阈值，且不在排除列中
    to_drop = nan_ratio[nan_ratio > threshold].index.difference(exclude_columns)

    print(f"删除的列数: {len(to_drop)}")
    print(f"被删除的列: {list(to_drop)}")
    
    # 删除列
    return df.drop(columns=to_drop)

if __name__ == "__main__":
    import data_load

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
    cleaned_df = drop_high_nan_columns(df)

    # 保存 删除缺失 后的数据
    cleaned_df.to_csv('../../raw_data/cleaned_data.csv', index=False)

    # 并按天聚合的数据
    df_daily = aggregate_by_day(cleaned_df)
    df_daily.to_csv("../../raw_data/cleaned_data_daily_summary.csv", index=False)
    print("df_daily data:", df_daily.isnull().mean().sort_values(ascending=False))        # 快速了解缺失情况

    # 聚合后，二次清洗mood为空的数据
    print(f"原始按天聚合数据共 {len(df_daily)} 条")
    df_daily = df_daily[df_daily['mood'].notna()].reset_index(drop=True)
    print(f"保留含有 mood 值的数据共 {len(df_daily)} 条")
    df_daily.to_csv("../../raw_data/cleaned_data_daily_summary_mood.csv", index=False)

    # 二次清理后，还剩下一些缺失情况
    print("df_daily data:", df_daily.isnull().mean().sort_values(ascending=False))        # 快速了解缺失情况

