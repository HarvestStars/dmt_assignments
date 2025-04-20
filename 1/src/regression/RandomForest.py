from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据（假设你已经有了训练和测试数据）
train_df = pd.read_csv("../classification/source_data/train_split_by_id.csv")
test_df = pd.read_csv("../classification/source_data/test_split_by_id.csv")

# 特征与目标
features = [col for col in train_df.columns if col.endswith('_hist_mean')]
target = "mood_target"

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 模型初始化 + 拟合
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# 预测
y_pred = reg.predict(X_test)

# 回归性能指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📉 MAE:", mae)
print("📉 MSE:", mse)
print("📈 R²:", r2)

# 假设你已经有了 y_test 和 y_pred（来自上面的代码）
plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test.values, label='True Mood', linewidth=2)
plt.plot(range(len(y_pred)), y_pred, label='Predicted Mood', linewidth=2, linestyle='--')
plt.title("Random Forest--True vs Predicted Mood (Sorted by Sample Order)", fontsize=16)
plt.xlabel("Sample Index", fontsize=16)
plt.ylabel("Mood", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../figs/RF_Mood.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x 参考线
plt.xlabel("True Mood", fontsize=16)
plt.ylabel("Predicted Mood", fontsize=16)
plt.title("Random Forest--True vs Predicted Mood (Scatter)", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../figs/RF_Mood_scatter.png", dpi=300)
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(10, 4))
plt.scatter(range(len(residuals)), residuals, alpha=0.6)
plt.hlines(0, 0, len(residuals), colors='r', linestyles='--')
plt.title("Random Forest--Residuals (True - Predicted)", fontsize=16)
plt.xlabel("Sample Index", fontsize=16)
plt.ylabel("Residual", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../figs/RF_Mood_residuals.png", dpi=300)
plt.show()
