import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

search_score = 5
click_score = 50
booking_score = 100

n_clusters = 20
# train_size = 2000000
train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv")

features = [
    'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
    'srch_children_count', 'srch_room_count', 'orig_destination_distance'
]
train_X = train_data[features]
train_X = imputer.fit_transform(train_X)
train_X = scaler.fit_transform(train_X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
train_data['search_cluster'] = kmeans.fit_predict(train_X)
clustered_unique = train_data.drop_duplicates(subset=['srch_id', 'search_cluster'])

user_hotel_scores = pd.read_csv(f'results/user_hotel_scores_{search_score}_{click_score}_{booking_score}.csv')
merged = pd.merge(clustered_unique, user_hotel_scores, on='srch_id', how='left')

cluster_id_lists = []
for cluster_num in range(n_clusters):
    cluster_rows = merged[merged['search_cluster'] == cluster_num]
    output = cluster_rows[['srch_id', 'prop_id_y', 'score']]
    n_unique_srch_id = output['srch_id'].nunique()
    output_sorted = output.sort_values('prop_id_y')
    score_sum = output.groupby('prop_id_y')['score'].sum()
    score_avg = score_sum / n_unique_srch_id
    score_avg_df = score_avg.reset_index().rename(columns={'score': 'average_score'})
    score_avg_df.to_csv(f"results/k_means_clusters/k_means_cluster_{n_clusters}_{cluster_num}.csv", index=False)

    srch_ids = cluster_rows['srch_id'].tolist()
    cluster_id_lists.append(srch_ids)

max_len = max(len(lst) for lst in cluster_id_lists)
cluster_id_lists_padded = [lst + [None]*(max_len - len(lst)) for lst in cluster_id_lists]
srch_id_clusters_df = pd.DataFrame(cluster_id_lists_padded).transpose()
srch_id_clusters_df.columns = [f'cluster_{i}' for i in range(n_clusters)]

unique_cols = {col: srch_id_clusters_df[col].drop_duplicates().tolist() for col in srch_id_clusters_df.columns}
max_len = max(len(vals) for vals in unique_cols.values())
for col in unique_cols:
    unique_cols[col] += [None] * (max_len - len(unique_cols[col]))

unique_df = pd.DataFrame(unique_cols)
unique_df.to_csv(f'results/k_means_clusters_{n_clusters}.csv', index=False)