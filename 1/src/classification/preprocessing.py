import pandas as pd
from sklearn.model_selection import train_test_split

# 加载原始特征数据（包括 mood_type）
df = pd.read_csv("../../raw_data/mood_classified_sliding_window.csv")

# 初始化训练和测试集 DataFrame
train_df = pd.DataFrame()
test_df = pd.DataFrame()

# 对每个 id 分别划分训练集和测试集（例如 70% 训练，30% 测试）
for pid in df['id'].unique():
    user_data = df[df['id'] == pid]
    user_train, user_test = train_test_split(user_data, test_size=0.3, random_state=42)
    train_df = pd.concat([train_df, user_train], ignore_index=True)
    test_df = pd.concat([test_df, user_test], ignore_index=True)

# 可选：保存训练集和测试集（用于后续模型调试）
train_df.to_csv("./source_data/train_split_by_id.csv", index=False)
test_df.to_csv("./source_data/test_split_by_id.csv", index=False)
