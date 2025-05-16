import numpy as np
import pandas as pd
from catboost import CatBoostRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# train_size = 10000
# train_data = pd.read_csv("../2/dmt-2025-2nd-assignment/training_set_VU_DM.csv", nrows=train_size)
train_data = pd.read_csv("../2/dmt-2025-2nd-assignment/training_set_VU_DM.csv")

conditions = [
    (train_data['booking_bool'] == 1),                     
    (train_data['click_bool'] == 1) & (train_data['booking_bool'] == 0)  
]
choices = [10, 5]
train_data['score'] = np.select(conditions, choices, default=0)

features = [
    'srch_id',
    'site_id',
    'visitor_location_country_id',
    'prop_country_id',
    'prop_id',
    'prop_starrating',
    'prop_review_score',
    'prop_brand_bool',
    'prop_location_score1',
    'prop_location_score2',
    'price_usd',
    'promotion_flag',
    'srch_destination_id',
    'srch_length_of_stay',
    'srch_booking_window',
    'srch_adults_count',
    'srch_children_count',
    'srch_room_count',
    'srch_saturday_night_bool'
]

# comp_features = [
#     'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff',
#     'comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff',
#     'comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff',
#     'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff',
#     'comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff',
#     'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff',
#     'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff',
#     'comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff'
# ]
# train_data[comp_features] = train_data[comp_features].fillna(0)

X = train_data[features]
y = train_data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X['srch_id']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

sample_weights = np.where(y == 10, 10, np.where(y == 5, 5, 1))

categorical_features = [
    'site_id',
    'visitor_location_country_id',
    'prop_country_id',
    'prop_id',  
    'srch_destination_id',
    'promotion_flag',
    'srch_saturday_night_bool'
]

# for col in categorical_features:
#     train_data[col] = train_data[col].astype(str).fillna('missing')
    
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=10,
    eval_metric='RMSE',
    cat_features=categorical_features,  # Specify categorical features
    early_stopping_rounds=100,  # Add early stopping
    random_seed=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    sample_weight=sample_weights[train_idx],
    verbose=100
)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")


# test_data = pd.read_csv("../2/dmt-2025-2nd-assignment/test_set_VU_DM.csv", nrows=train_size)
# test_data = pd.read_csv("../2/dmt-2025-2nd-assignment/test_set_VU_DM.csv")

# X_test = test_data[features]
# test_predictions = model.predict(X_test)
# test_data['predicted_score'] = test_predictions
# test_data = test_data[['srch_id', 'prop_id', 'predicted_score']]
# sorted_data = test_data.sort_values(
#     by=['srch_id', 'predicted_score'],
#     ascending=[True, False]
# ).reset_index(drop=True)
# sorted_data[['srch_id','prop_id']].to_csv("catboost_predictions_iter_5000.csv", index=False)