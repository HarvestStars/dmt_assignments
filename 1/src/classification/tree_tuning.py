# tuning parameters decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
param_grid = {
    "max_depth": [3, 5, 7, 9, 11, 15, None],  # Try various depths
    "min_samples_leaf": [1, 3, 5, 7, 9],  # Optional: try leaf constraints too
}
tree = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all CPU cores
    verbose=1,
)

filepath = "../../raw_data/cleaned_data_daily_summary_mood_imputed_sliding_window.csv"
df = pd.read_csv(filepath)
df.fillna(0, inplace=True)
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

scaler = StandardScaler()  # standardize the data
X_scaled = scaler.fit_transform(X)  # fit the scaler to the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, train_size=0.90, random_state=42
)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
print(f"Test RMSE from best model: {rmse:.4f}")
