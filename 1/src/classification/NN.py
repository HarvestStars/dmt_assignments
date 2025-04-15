import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import openai as gpt

# ========== 读取数据并填充缺失 ==========
filepath = "../../raw_data/cleaned_data_daily_summary_mood_imputed_sliding_window.csv"
df = pd.read_csv(filepath)
df.fillna(0, inplace=True)

# ========== 特征和目标 ==========
features = [
    'mood_hist_mean', 'activity_hist_mean',
    'appCat.builtin_hist_mean', 'appCat.communication_hist_mean',
    'appCat.entertainment_hist_mean', 'appCat.social_hist_mean',
    'screen_hist_mean'
]
X = df[features].values
y = df['mood_target'].values

# ========== 标准化 + 划分训练测试集 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== 转为 Tensor ==========
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ========== 构建神经网络回归模型 ==========
class MoodRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MoodRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ========== 训练 ==========
for epoch in range(200):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

# ========== 评估 ==========
model.eval()
with torch.no_grad():
    pred = model(X_test_tensor).squeeze().numpy()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"\n🧠 NN 回归结果：MSE={mse:.4f}, R²={r2:.4f}")
