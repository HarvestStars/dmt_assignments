import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
from pathlib import Path

from data_impute_and_fe.process_A_srch import process_search_features_smoothed
from data_impute_and_fe.process_B_prop import process_hotel_features
from data_impute_and_fe.process_C_price import process_price_feature_smoothed
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
df = pd.read_csv(CSV_PATH, nrows=5_000_000)

df_final, cols_A, cols_categorical_A = process_search_features_smoothed(df, drop_raw_columns=False)
print(f"Processed A class features: {cols_A}")

df_final, cols_B, cols_categorical_B = process_hotel_features(df_final, drop_raw_columns=False)
print(f"Processed B class features: {cols_B}")

df_df_finalc_clean, cols_C, cols_categorical_C = process_price_feature_smoothed(df_final, drop_raw_columns=False)
print(f"Processed C class features: {cols_C}")

df_final, cols_D, cols_categorical_D = process_new_user(df_final, drop_raw_columns=False)
print(f"Processed D class features: {cols_D}")

# df_final, cols_E, cols_categorical_E = process_distance_feature(df_final, drop_raw_columns=False)
# print(f"Processed E class features: {cols_E}")
cols_E = []
cols_categorical_E = []

# df_final, cols_F, cols_categorical_F = process_random_feature(df_final, drop_raw_columns=False)
# print(f"Processed F class features: {cols_F}")
cols_F = []
cols_categorical_F = []

# df_final, cols_G, cols_categorical_G = add_competition_features(df_final, drop_raw_columns=False)
# print(f"Processed G class features: {cols_G}")
cols_G = []
cols_categorical_G = []

df_final, cols_train_label = process_train_label(df_final, drop_raw_columns=False)
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

# ========== Step 2: 编码分类变量 ==========
print("[1/6] Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ========== Step 3: 拆分 train / valid（按 search_id 分组） ==========
print("[2/6] Splitting train/validation set...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups=df_final["srch_id"]))

X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

train_srch_ids = df_final.iloc[train_idx]["srch_id"]
valid_srch_ids = df_final.iloc[valid_idx]["srch_id"]

group_train = train_srch_ids.value_counts().loc[train_srch_ids.unique()].tolist()
group_valid = valid_srch_ids.value_counts().loc[valid_srch_ids.unique()].tolist()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_train)

dvalid = xgb.DMatrix(X_valid, label=y_valid)
dvalid.set_group(group_valid)

# ========== Step 4: 训练 XGBoost 排序模型 ==========
print("[3/6] Training XGBoost...")
params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "max_depth": 6,
    "verbosity": 1,
    "eval_metric": "ndcg@5",
    "tree_method": "hist",
}
bst = xgb.train(params, dtrain, num_boost_round=100)

# ========== Step 5: 预测验证集，计算 NDCG@5 ==========
print("[4/6] Validating NDCG@5...")
X_valid_copy = X_valid.copy()
X_valid_copy["score"] = bst.predict(dvalid)
X_valid_copy["train_relevance"] = y_valid
X_valid_copy["srch_id"] = valid_srch_ids.values

ndcg_scores = []
for srch_id, group in X_valid_copy.groupby("srch_id"):
    y_true = group["train_relevance"].values.reshape(1, -1)
    y_score = group["score"].values.reshape(1, -1)
    if y_true.shape[1] >= 2:  # 至少两个候选酒店才能计算排名
        ndcg = ndcg_score(y_true, y_score, k=5)
        ndcg_scores.append(ndcg)

avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
print(f"[5/6] Avg NDCG@5 on validation set: {avg_ndcg:.5f}")

# ========== Step 6: 保存模型 ==========
MODEL_PATH = Path("models") / "xgb_rank_model.json"
MODEL_PATH.parent.mkdir(exist_ok=True)
bst.save_model(str(MODEL_PATH))
print(f"[6/6] Model saved to {MODEL_PATH}")