import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

# ========== Step 1: 读取数据并处理 ==========
print("[1/6] Reading and processing raw data...")
CSV_PATH = "./dmt-2025-2nd-assignment/training_set_VU_DM.csv"
OUT_DIR = Path("split_outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH, nrows=1_000_000)

df_final, cols_A, cols_categorical_A = process_search_behavior_features(df)
print(f"Processed A class features: {cols_A}")

df_final, cols_B, cols_categorical_B = process_hotel_features(df_final)
print(f"Processed B class features: {cols_B}")

df_final, cols_C, cols_categorical_C = process_price_feature(df_final)
print(f"Processed C class features: {cols_C}")

# df_final, cols_D, cols_categorical_D = process_new_user(df_final)
# print(f"Processed D class features: {cols_D}")
cols_D = []
cols_categorical_D = []

df_final, cols_E, cols_categorical_E = process_distance_feature(df_final)
print(f"Processed E class features: {cols_E}")

df_final, cols_F, cols_categorical_F = process_random_feature(df_final)
print(f"Processed F class features: {cols_F}")

# df_final, cols_G, cols_categorical_G = add_competition_features(df_final)
# print(f"Processed G class features: {cols_G}")

df_final, cols_train_label = process_train_label(df_final)
print(f"Processed train label class features: {cols_train_label}")

feature_cols = cols_A + cols_B + cols_C + cols_D + cols_E + cols_F #+ cols_G
categorical_cols = (
    cols_categorical_A + cols_categorical_B + cols_categorical_C +
    cols_categorical_D + cols_categorical_E + cols_categorical_F #+ cols_categorical_G
)

# ========== Step 2: 编码分类特征 ==========
print("[2/6] Encoding categorical features...")
for col in categorical_cols:
    if col in df_final.columns:
        df_final[col] = pd.Categorical(df_final[col]).codes

X = df_final[feature_cols].values.astype(np.float32)
y = df_final[cols_train_label].values.astype(np.float32).flatten()

if np.isnan(X).any() or np.isinf(X).any():
    print("⚠️ X contains NaN or Inf values")
    nan_cols = [col for col in feature_cols if df_final[col].isnull().any()]
    print(f"Columns with NaNs: {nan_cols}")

# ========== Step 3: 创建 Dataset 和 DataLoader ==========
class BookingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 拆分训练验证集
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(df_final, groups=df_final['srch_id']))

train_ds = BookingDataset(X[train_idx], y[train_idx])
valid_ds = BookingDataset(X[valid_idx], y[valid_idx])

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=1024)

# ========== Step 4: 定义神经网络结构 ==========
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),  # 输出打分
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

model = SimpleNN(input_dim=X.shape[1])

# ========== Step 5: 训练模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # 用于Pointwise回归目标

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss / len(train_ds):.4f}")

# ========== Step 6: 验证 NDCG@5 ==========
model.eval()
valid_preds = []
valid_labels = []
valid_srch_ids = df_final.iloc[valid_idx]['srch_id'].values

with torch.no_grad():
    for xb, yb in valid_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        valid_preds.extend(preds)
        valid_labels.extend(yb.numpy())

# 构造 DataFrame 便于 groupby
valid_df = pd.DataFrame({
    'srch_id': valid_srch_ids,
    'score': valid_preds,
    'train_relevance': valid_labels
})

ndcg_scores = []
for srch_id, group in valid_df.groupby('srch_id'):
    y_true = group['train_relevance'].values.reshape(1, -1)
    y_score = group['score'].values.reshape(1, -1)
    if y_true.shape[1] >= 2:
        ndcg = ndcg_score(y_true, y_score, k=5)
        ndcg_scores.append(ndcg)

print(f"\n✅ Mean NDCG@5 on validation set: {np.mean(ndcg_scores):.4f}")
