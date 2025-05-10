import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from data_impute_and_fe.process_A_srch import process_search_features_smoothed
from data_impute_and_fe.process_B_prop import process_hotel_features
from data_impute_and_fe.process_C_price import process_price_feature_smoothed
from data_impute_and_fe.process_D_user import process_new_user
from data_impute_and_fe.process_E_distance import process_distance_feature
from data_impute_and_fe.process_F_random import process_random_feature
from data_impute_and_fe.process_G_comp import add_competition_features
from data_impute_and_fe.process_train_label import process_train_label

# ========== Step 1: 路径与文件 ==========
MODEL_PATH = Path("models/xgb_rank_model.json")
TEST_CSV = "./dmt-2025-2nd-assignment/test_set_VU_DM.csv"
SUBMIT_CSV = "submission.csv"

# ========== Step 2: 加载模型 ==========
print("[1/5] Loading trained model...")
bst = xgb.Booster()
bst.load_model(str(MODEL_PATH))

# ========== Step 3: 加载并预处理测试集 ==========
print("[2/5] Loading and processing test set...")
df_test_raw = pd.read_csv(TEST_CSV)

# 👇 这里用你之前的预处理函数（保持和训练一致）
df_test, cols_A, cols_categorical_A = process_search_features_smoothed(df_test_raw)
df_test, cols_B, cols_categorical_B = process_hotel_features(df_test)
df_test, cols_C, cols_categorical_C = process_price_feature_smoothed(df_test)
df_test, cols_D, cols_categorical_D = process_new_user(df_test)
# df_test, _, _ = process_distance_feature(df_test)
# df_test, _, _ = process_random_feature(df_test)
# df_test, _, _ = add_competition_features(df_test)

# 特征列（必须与训练一致）

feature_cols = (
    cols_A + cols_B + cols_C  + cols_D
)

categorical_cols = (
    cols_categorical_A + cols_categorical_B + cols_categorical_C+ cols_categorical_D
)

X_test = df_test[feature_cols].copy()

# ========== Step 4: 分类变量编码 ==========
print("[3/5] Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))  # 防止 NaN

# ========== Step 5: 模型预测 ==========
print("[4/5] Predicting scores...")
dtest = xgb.DMatrix(X_test)
df_test["score"] = bst.predict(dtest)

# ========== Step 6: 输出提交文件 ==========
print("[5/5] Generating submission file...")

submission_df = (
    df_test[["srch_id", "prop_id", "score"]]
    .sort_values(["srch_id", "score"], ascending=[True, False])
    .groupby("srch_id")[["prop_id"]]
    .apply(lambda x: x.astype(int))
    .reset_index()
    .drop(columns=["level_1"])
)

submission_df.columns = ["srch_id", "prop_id"]
submission_df.to_csv(SUBMIT_CSV, index=False)
print(f"✅ Submission file saved to {SUBMIT_CSV}")
