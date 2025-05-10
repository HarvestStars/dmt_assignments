import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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

# ========== Step 1: 读取数据并处理 ==========
print("[1/6] Reading and processing raw data...")
CSV_PATH = "./dmt-2025-2nd-assignment/training_set_VU_DM.csv"
OUT_DIR = Path("split_outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, nrows=100_000)

df_final, cols_A, cols_categorical_A = process_search_behavior_features(df)
print(f"Processed A class features: {cols_A}")

df_final, cols_B, cols_categorical_B = process_hotel_features(df_final)
print(f"Processed B class features: {cols_B}")

df_final, cols_C, cols_categorical_C = process_price_feature(df_final)
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

feature_cols = cols_A + cols_B + cols_C + cols_D + cols_E + cols_F + cols_G
categorical_cols = (
    cols_categorical_A + cols_categorical_B + cols_categorical_C +
    cols_categorical_D + cols_categorical_E + cols_categorical_F + cols_categorical_G
)

X = df_final[feature_cols].copy()
y = df_final[cols_train_label].copy()

# ========== Step 2: 转换分类变量为类别编码（Label Encoding） ==========
print("[2/6] Encoding categorical features...")
for col in categorical_cols:
    if col in X.columns:
        X[col] = pd.Categorical(X[col]).codes

# ========== Step 3: 拆分训练集和验证集 ==========
print("[3/6] Splitting data into train and validation sets...")
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========== Step 4: 训练 XGBoost 分类器 ==========
print("[4/6] Training XGBoost classifier...")
model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    tree_method='hist',
    eval_metric='mlogloss',
    use_label_encoder=False,
    verbosity=1
)

model.fit(train_X, train_y.values.ravel())

# ========== Step 5: 验证集预测并计算 NDCG@5 ==========
print("[5/6] Predicting and evaluating NDCG@5...")
valid_df = valid_X.copy()
valid_df['srch_id'] = df_final.loc[valid_X.index, 'srch_id'].values
valid_df['train_relevance'] = valid_y.values

# 取出预测的 booking (label=2) 概率作为 score
proba = model.predict_proba(valid_X)
valid_df['score'] = proba[:, 2]

# 按 srch_id 分组计算 NDCG@5
ndcg_scores = []
for srch_id, group in valid_df.groupby('srch_id'):
    y_true = group['train_relevance'].values.reshape(1, -1)
    y_score = group['score'].values.reshape(1, -1)
    if y_true.shape[1] >= 2:
        ndcg = ndcg_score(y_true, y_score, k=5)
        ndcg_scores.append(ndcg)

print(f"\n✅ Mean NDCG@5 on validation set: {np.mean(ndcg_scores):.4f}")
