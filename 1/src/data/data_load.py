import pandas as pd

def load_mood_dataset(filepath):
    """
    读取原始长格式数据并转换为用户-时间-变量的多列宽格式。
    
    参数:
        filepath: str - CSV/Excel 路径
        
    返回:
        df_pivot: DataFrame - 每个时间点、每个用户的变量在一行显示
    """
    # Step 1: load data
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)

    # Step 2: preprocess
    df.columns = df.columns.str.lower()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['id'] = df['id'].astype(str)
    df['variable'] = df['variable'].astype(str)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Step 3: wide format
    df_pivot = df.pivot_table(
        index=['id', 'time'],
        columns='variable',
        values='value',
        aggfunc='mean'  # mean
    ).reset_index()

    # time ordering
    df_pivot = df_pivot.sort_values(by=['id', 'time']).reset_index(drop=True)

    return df_pivot

def export_to_csv(df, output_path):
    """
    将 DataFrame 导出为 CSV 文件。
    
    参数:
        df: DataFrame - 要保存的数据表
        output_path: str - 导出路径，如 'output.csv'
    """
    df.to_csv(output_path, index=False)
    print(f"已成功导出到：{output_path}")

def inspect_random_variable_column(df, column_name=None):
    """
    随机选择一个非 ['id', 'time'] 的列，统计其非空值数量。
    
    参数:
        df: DataFrame - 已拉平的数据表
    
    返回:
        column_name: str - 被选中的列名
        non_null_count: int - 非空值数量
    """
    variable_columns = [col for col in df.columns if col not in ['id', 'time']]
    if not variable_columns:
        print("没有可用的变量列。")
        return None, 0
    
    if column_name == None:
        column_name = random.choice(variable_columns)
    non_null_count = df[column_name].notna().sum()
    
    print(f"随机选中的列: {column_name}")
    print(f"非空值数量: {non_null_count}")
    
    return column_name, non_null_count


if __name__ == "__main__":
    import random
    # load data
    df = load_mood_dataset("../../raw_data/dataset_mood_smartphone.csv")
    print(df.head(3))

    # 快速了解缺失情况
    print(df.isnull().mean().sort_values(ascending=False))

    # 随机选择一个变量列并统计非空值数量
    column_name, non_null_count = inspect_random_variable_column(df, column_name="appCat.entertainment")
    print(f"随机选中的列: {column_name}, 非空值数量: {non_null_count}")

    # 导出数据
    export_to_csv(df, "../../raw_data/dataset_mood_smartphone_wide.csv")
