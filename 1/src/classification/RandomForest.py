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
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low (<7)", "High (≥7)"])
disp.plot(cmap="Blues", values_format='d')

# 评估指标
print("Accuracy:", accuracy_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Low (<7)", "High (≥7)"]))

# 统计各类别数量（真实和预测）
actual_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()

# 确保两个 Series 有相同索引（0 和 1）
all_classes = [0, 1]
actual_counts = actual_counts.reindex(all_classes, fill_value=0)
pred_counts = pred_counts.reindex(all_classes, fill_value=0)

# 合并到一个 DataFrame
compare_df = pd.DataFrame({
    'Actual': actual_counts,
    'Predicted': pred_counts
})

# 绘图
compare_df.plot(kind='bar', figsize=(7, 5), color=['orange', 'steelblue'])
plt.title("Random Forest \nActual vs Predicted Class Counts (Test Set)", fontsize=16)
plt.xlabel("Mood Type (Binary)", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.xticks(ticks=[0, 1], labels=["Low (<7)", "High (≥7)"], rotation=0, fontsize=14)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("../../figs/actual_vs_predicted_counts.png", dpi=300)
plt.show()