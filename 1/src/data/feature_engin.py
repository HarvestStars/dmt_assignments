import pandas as pd
from datetime import timedelta

def create_sliding_window_dataset(df: pd.DataFrame, window_size=5):
    """
    使用滑动窗口创建输入-输出对（X, y），每条记录按用户单独处理。
    仅保留时间连续的窗口（天数不允许间断）。
    
    参数:
    - df: 清洗后的 DataFrame，包含 ['id', 'date', ..., 'mood']
    - window_size: 滑动窗口大小，默认为 5
    
    返回:
    - df_windows: 新的训练集 DataFrame，包含均值特征和第6天 mood 为 target
    """
    df = df.copy()
    df = df.sort_values(by=['id', 'date'])

    result_rows = []

    feature_cols = df.columns.difference(['id', 'date', 'mood'])

    for user_id, group in df.groupby('id'):
        group = group.reset_index(drop=True)

        for i in range(window_size, len(group)):
            window = group.iloc[i - window_size:i + 1]  # 含第6天

            # 检查是否是连续的6天（时间差是 [1,1,1,1,1]）
            date_diffs = [
                (window.iloc[j]['date'] - window.iloc[j - 1]['date']).days
                for j in range(1, window_size + 1)
            ]
            if all(d == 1 for d in date_diffs):
                hist_window = window.iloc[:window_size]
                target_day = window.iloc[window_size]

                feature_means = hist_window[feature_cols].mean()
                mood_mean = hist_window['mood'].mean()

                row_data = {
                    'id': user_id,
                    'date': target_day['date'],
                    'mood_target': target_day['mood'],
                    'mood_hist_mean': mood_mean,
                }

                # 加入历史特征均值
                for col in feature_cols:
                    row_data[f'{col}_hist_mean'] = feature_means[col]

                result_rows.append(row_data)

    df_windows = pd.DataFrame(result_rows)
    return df_windows

if __name__ == "__main__":
    # 加载数据
    filepath = "../../raw_data/cleaned_data_daily_summary_mood_imputed.csv"
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    df['date'] = pd.to_datetime(df['date'])  # 确保是 datetime 类型
    df_features = create_sliding_window_dataset(df, window_size=5)

    print(df_features.head())
    print("构造后的样本总数:", len(df_features))

    df_features.to_csv("../../raw_data/cleaned_data_daily_summary_mood_imputed_sliding_window.csv", index=False)
