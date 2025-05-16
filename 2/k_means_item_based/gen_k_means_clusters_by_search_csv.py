import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

search_score = 10
click_score = 100
booking_score = 500

n_clusters = 40
# train_size = 1000
# train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv", nrows=train_size)
train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv")

# features = [
#     'prop_starrating', 'prop_review_score', 'prop_brand_bool',
#     'prop_location_score1', 'prop_location_score2',
#     'prop_log_historical_price', 'price_usd', 'promotion_flag',
#     'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
#     'srch_children_count', 'srch_room_count', 'orig_destination_distance'
# ]

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
    'position',
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

for cluster_id in range(n_clusters):
    cluster_df = train_data[train_data['search_cluster'] == cluster_id].copy()
    
    cluster_df['score'] = search_score 
    cluster_df.loc[(cluster_df['click_bool'] == 1) & (cluster_df['booking_bool'] == 0), 'score'] = click_score
    cluster_df.loc[cluster_df['booking_bool'] == 1, 'score'] = booking_score
    
    cluster_df = cluster_df[['prop_id', 'score']]
    cluster_df = cluster_df.groupby('prop_id', as_index=False)['score'].sum()
    cluster_df = cluster_df.sort_values(by='score', ascending=False)
    
    filename = f"results/k_means_clusters_by_search/k_means_cluster_by_search_feature_3_{cluster_id}.csv"
    cluster_df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(cluster_df)} rows.")
