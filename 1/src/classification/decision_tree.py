# Apply decision tree for regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


"""
Citation site: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
"""
# step 1: load the data first
filepath = "../../raw_data/cleaned_data_daily_summary_mood_imputed_sliding_window.csv"
df = pd.read_csv(filepath)
df.fillna(0, inplace=True)

# step 2: group the predictors and targets
features = [
    "mood_hist_mean",
    "activity_hist_mean",
    "appCat.builtin_hist_mean",
    "appCat.office_hist_mean",
    "call_hist_mean",
    "circumplex.arousal_hist_mean",
    "circumplex.valence_hist_mean",
    "screen_hist_mean",
]
X = df[features].values
y = df["mood_target"].values

# step 3: transform the data and divide the dataset into training and testing sets
scaler = StandardScaler()  # standardize the data
X_scaled = scaler.fit_transform(X)  # fit the scaler to the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, train_size=0.90, random_state=42
)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# step 4: build the decision tree models
model1 = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=5,  # Don't split if a leaf has fewer than 5 samples
)
model1.fit(X_train, y_train)
train_predictions = model1.predict(X_train)
test_predictions = model1.predict(X_test)

model2 = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=5,  # Don't split if a leaf has fewer than 5 samples
)
model2.fit(X_train, y_train)
train_predictions2 = model2.predict(X_train)
test_predictions2 = model2.predict(X_test)

# step 5: evaluate the model
mse_train = mean_squared_error(y_train, train_predictions)
mse_test = mean_squared_error(y_test, test_predictions)
mse_train2 = mean_squared_error(y_train, train_predictions2)
mse_test2 = mean_squared_error(y_test, test_predictions2)

mae_train = mean_absolute_error(y_train, train_predictions)
mae_test = mean_absolute_error(y_test, test_predictions)
mae_train2 = mean_absolute_error(y_train, train_predictions2)
mae_test2 = mean_absolute_error(y_test, test_predictions2)

print(f"Train MSE (max_depth=3): {mse_train:.4f}")
print(f"Test MSE (max_depth=3): {mse_test:.4f}")
print(f"Train MSE (max_depth=5): {mse_train2:.4f}")
print(f"Test MSE (max_depth=5): {mse_test2:.4f}")

print(f"Train MAE (max_depth=3): {mae_train:.4f}")
print(f"Test MAE (max_depth=3): {mae_test:.4f}")
print(f"Train MAE (max_depth=5): {mae_train2:.4f}")
print(f"Test MAE (max_depth=5): {mae_test2:.4f}")

# step 6: plot the results
plt.figure(figsize=(10, 5))
plt.plot(y_train, label="True Values", color="red", linestyle="--")
plt.plot(train_predictions, label="Predictions (max_depth=3)", color="blue")
plt.plot(train_predictions2, label="Predictions (max_depth=5)", color="green")
plt.legend()
plt.grid()
plt.title("Decision Tree Regression Predictions (train set)", fontsize=16)
plt.xlabel("Test set", fontsize=14)
plt.ylabel("Target mood", fontsize=14)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="True Values", color="red", linestyle="--")
plt.plot(test_predictions, color="blue", marker="o")
plt.plot(test_predictions2, color="green", marker="o")
plt.legend()
plt.grid()
plt.title("Decision Tree Regression Predictions (test set)", fontsize=22)
plt.xlabel("Test set", fontsize=20)
plt.ylabel("Target mood", fontsize=20)
plt.legend(
    ["True values", "Predictions(max_depth=3)", "Predictions (max_depth=5)"],
    fontsize="14",
    loc="lower right",
)
plt.show()

# step 7: plot the decision tree
plt.figure(figsize=(20, 10))
feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
plot_tree(
    model1,  # ✅ Your fitted DecisionTreeClassifier or Regressor
    feature_names=feature_names,  # ✅ Replace with full list of feature names
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree Structure")
plt.show()
