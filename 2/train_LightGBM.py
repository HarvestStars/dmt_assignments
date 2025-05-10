import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
from pathlib import Path

from data_impute_and_fe.process_A_srch import process_search_behavior_features
from data_impute_and_fe.process_B_prop import process_hotel_features
from data_impute_and_fe.process_C_price import process_price_feature
from data_impute_and_fe.process_D_user import process_new_user
from data_impute_and_fe.process_E_distance import process_distance_feature
from data_impute_and_fe.process_F_random import process_random_feature
from data_impute_and_fe.process_G_comp import add_competition_features
from data_impute_and_fe.process_train_label import process_train_label


# Features 和 Label
CSV_PATH = "./dmt-2025-2nd-assignment/training_set_VU_DM.csv"
OUT_DIR = Path("split_outputs")
OUT_DIR.mkdir(exist_ok=True)


# ========== Step 1: 读取数据并处理 ==========
print("[1/6] Reading and processing raw data...")
df = pd.read_csv(CSV_PATH, nrows=1_000_000)

df_final, cols_A, cols_categorical_A = process_search_behavior_features(df)
print(f"Processed A class features: {cols_A}")

df_final, cols_B, cols_categorical_B = process_hotel_features(df_final)
print(f"Processed B class features: {cols_B}")

df_df_finalc_clean, cols_C, cols_categorical_C = process_price_feature(df_final)
print(f"Processed C class features: {cols_C}")

df_final, cols_D, cols_categorical_D = process_new_user(df_final)
print(f"Processed D class features: {cols_D}")

df_final, cols_E, cols_categorical_E = process_distance_feature(df_final)
print(f"Processed E class features: {cols_E}")

df_final, cols_F, cols_categorical_F = process_random_feature(df_final)
print(f"Processed F class features: {cols_F}")

df_final, cols_G, cols_categorical_G = add_competition_features(df_final)
print(f"Processed G class features: {cols_G}")

df_final, cols_train_label = process_train_label(df_final)
print(f"Processed train label class features: {cols_train_label}")

feature_cols = (
    cols_A + cols_B + cols_C  + cols_D + cols_E + cols_F +cols_G
)

categorical_cols = (
    cols_categorical_A + cols_categorical_B + cols_categorical_C+ cols_categorical_D + 
    cols_categorical_E + cols_categorical_F + cols_categorical_G
)

X = df_final[feature_cols]
y = df_final[cols_train_label]

# ========== Step 2: 构造训练集和验证集 ==========
print("[2/6] Splitting into train and validation sets...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(df_final, groups=df_final['srch_id']))

train_df = df_final.iloc[train_idx].copy()
valid_df = df_final.iloc[valid_idx].copy()

# group 数组：每个 query 的文档数量（按 srch_id 分组）
train_group = train_df.groupby('srch_id').size().to_list()
valid_group = valid_df.groupby('srch_id').size().to_list()

# ========== Step 3: 构造 LightGBM Datasets ==========
print("[3/6] Preparing LightGBM datasets...")
for col in categorical_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
        valid_df[col] = valid_df[col].astype('category')

lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[cols_train_label], group=train_group, categorical_feature = categorical_cols)
lgb_valid = lgb.Dataset(valid_df[feature_cols], label=valid_df[cols_train_label], group=valid_group, categorical_feature = categorical_cols)

# ========== Step 4: 设置模型参数 ==========
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbosity": -1,
}

# ========== Step 5: 训练模型 ==========
print("[4/6] Training LightGBM model...")
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train', 'valid'],
    num_boost_round=200
)

# ========== Step 6: 验证集 NDCG@5 ==========
# 对每个 srch_id 分别预测评分并计算 NDCG
print("[5/6] Predicting on validation set...")
valid_df['score'] = model.predict(valid_df[feature_cols])

# 准备 NDCG 计算结构
print("[6/6] Calculating NDCG@5...")
ndcg_scores = []
for srch_id, group in valid_df.groupby('srch_id'):
    true_relevance = group[cols_train_label].values.reshape(1, -1)
    pred_scores = group['score'].values.reshape(1, -1)
    if len(true_relevance[0]) >= 2:
        ndcg = ndcg_score(true_relevance, pred_scores, k=5)
        ndcg_scores.append(ndcg)

print(f"Mean NDCG@5 on validation set: {np.mean(ndcg_scores):.4f}")