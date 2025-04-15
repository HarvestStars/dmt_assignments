import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Step 1: 读取数据
# ==========================
filepath = "../../raw_data/cleaned_data_daily_summary_mood_imputed_sliding_window.csv"
df = pd.read_csv(filepath)

# ==========================
# Step 2: 处理缺失值
# ==========================
if df.isnull().values.any():
    print("⚠️ 检测到缺失值，自动填充为 0")
    df.fillna(0, inplace=True)

# ==========================
# Step 3: 离散化目标变量 mood_target 为分类变量
# ==========================
def discretize_mood(mood):
    if mood < 6:
        return 0  # 低
    elif mood <= 7:
        return 1  # 中
    else:
        return 2  # 高

df['mood_class'] = df['mood_target'].apply(discretize_mood)

# ==========================
# Step 4: 特征提取
# ==========================
features = [
    'mood_hist_mean', 'activity_hist_mean',
    'appCat.builtin_hist_mean', 'appCat.communication_hist_mean',
    'appCat.entertainment_hist_mean', 'appCat.social_hist_mean',
    'screen_hist_mean'
]
X = df[features].values
y = df['mood_class'].values

# ==========================
# Step 5: 标准化 + 分割数据集
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
clf = SVC(kernel='rbf', C=1.0, gamma='scale')  # 可调 kernel、C、gamma
clf.fit(X_train, y_train)

# ==========================
# Step 6: 预测 & 评估
# ==========================
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM 分类准确率: {accuracy:.2f}")

print("\n详细分类报告：")
print(classification_report(y_test, y_pred, target_names=["低", "中", "高"]))