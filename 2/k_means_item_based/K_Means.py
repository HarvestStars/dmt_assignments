import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from utils import get_result, round_to_0_1_5, choose_k_for_KMeans

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

train_size = 1_000_00
test_size = 1_000_00
n_clusters = 100
train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv", nrows=train_size)

# Feature selection for KMeans
features = [
    'prop_starrating', 'prop_review_score', 'prop_brand_bool',
    'prop_location_score1', 'prop_location_score2',
    'prop_log_historical_price', 'price_usd', 'promotion_flag',
    'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
    'srch_children_count', 'srch_room_count', 'orig_destination_distance'
]
train_X = train_data[features]

# Imputation with mean
train_X = imputer.fit_transform(train_X)
train_X = scaler.fit_transform(train_X)

# # (optional) Choose the optimal k by elbow method
# choose_k_for_KMeans(train_X, 100)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
train_data['search_cluster'] = kmeans.fit_predict(train_X)

train_data['result'] = train_data.apply(get_result, axis=1)

# 7. Compute mean result per cluster
cluster_mean_result = train_data.groupby('search_cluster')['result'].mean()
cluster_mean_result = train_data.groupby('search_cluster')['result'].mean().apply(round_to_0_1_5)

# Read the next test_size rows after train_size
test_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv",
                        skiprows=range(1, train_size+1), nrows=test_size)

test_X = test_data[features]

test_X_imputed = imputer.transform(test_X)  # Imputation
test_X_scaled = scaler.transform(test_X_imputed)  # Scaling

test_clusters = kmeans.predict(test_X_scaled) # Use k-means
predicted_results = [cluster_mean_result[cluster] for cluster in test_clusters]

# for i, res in enumerate(predicted_results):
#     print(f"Predicted result for new search {i} (cluster {test_clusters[i]}): {res}")

test_data['result'] = test_data.apply(get_result, axis=1)
comparison_df = pd.DataFrame({
    'actual_result': test_data['result'].values,
    'predicted_result': predicted_results
})

accuracy = (comparison_df['actual_result'] == comparison_df['predicted_result']).mean()
mse = ((comparison_df['actual_result'] - comparison_df['predicted_result']) ** 2).mean()
print(f"Exact match accuracy: {accuracy:.2%}")
# print(f"Mean Squared Error: {mse:.4f}")
