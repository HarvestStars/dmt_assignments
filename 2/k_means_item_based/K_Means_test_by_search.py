import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

n_clusters = 40
# train_size = 1000
# train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv", nrows=train_size)
train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv")

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

train_X = train_data[features]
train_X = imputer.fit_transform(train_X)
train_X = scaler.fit_transform(train_X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
train_data['search_cluster'] = kmeans.fit_predict(train_X)

# new_data = pd.read_csv("../dmt-2025-2nd-assignment/test_set_VU_DM.csv", nrows=train_size)
new_data = pd.read_csv("../dmt-2025-2nd-assignment/test_set_VU_DM.csv")
new_X = new_data[features]
new_X = imputer.fit_transform(new_X)
new_X = scaler.fit_transform(new_X)
new_data['search_cluster'] = kmeans.predict(new_X)

new_data = new_data[['srch_id', 'prop_id', 'search_cluster']]
new_data['score'] = 0

cluster_dict = {}
for n in range(n_clusters):
    filename = f'results/k_means_clusters_by_search/k_means_cluster_by_search_feature_3_{n}.csv'
    df = pd.read_csv(filename)
    cluster_dict[n] = df.set_index('prop_id')['score'].to_dict()

def get_average_score(row):
    cluster = row['search_cluster']
    prop_id = row['prop_id']
    return cluster_dict.get(cluster, {}).get(prop_id, 0)

new_data['score'] = new_data.apply(lambda row: row['score'] + get_average_score(row), axis=1)
new_data_sorted = new_data.sort_values(
    by=['srch_id', 'score'],
    ascending=[True, False]
).reset_index(drop=True)
new_data_sorted = new_data_sorted[['srch_id', 'prop_id']]
new_data_sorted.to_csv(f'results/prediction_by_search_k_{n_clusters}_feature_3.csv', index=False)
