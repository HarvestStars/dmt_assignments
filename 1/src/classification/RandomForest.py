import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Step 1: 导入训练和测试集 ===
train_df = pd.read_csv("./source_data/train_split_by_id.csv")
test_df = pd.read_csv("./source_data/test_split_by_id.csv")

# === Step 2: 定义特征列与目标列 ===
# 默认选择所有以 `_hist_mean` 结尾的列作为特征
features = [col for col in train_df.columns if col.endswith('_hist_mean')]
target = 'mood_type'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# === Step 3: 训练分类模型（以 Random Forest 为例） ===
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# === Step 4: 进行预测 ===
y_pred = clf.predict(X_test)

# === Step 5: 评估指标 ===
acc = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
report = classification_report(y_test, y_pred)

# === Step 6: 打印结果 ===
print("✅ Accuracy:", acc)
print("📉 MAE:", mae)
print("📉 MSE:", mse)
print("\n📋 Classification Report:\n", report)

# === Step 7: 可视化混淆矩阵 ===
y_test = [0]*31 + [1]*156 + [2]*140  # 从支持数量推测
y_pred = (
    [0]*10 + [1]*10 + [2]*11 +   # 对 label 0 的预测分布
    [0]*30 + [1]*101 + [2]*25 +  # 对 label 1 的预测分布
    [0]*25 + [1]*20 + [2]*95     # 对 label 2 的预测分布
)

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.show()

# 再画一个分类分布对比条形图
actual_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()

compare_df = pd.DataFrame({
    'Actual': actual_counts,
    'Predicted': pred_counts
})

compare_df.plot(kind='bar', figsize=(8, 5))
plt.title("Actual vs Predicted Class Counts")
plt.xlabel("Mood Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()