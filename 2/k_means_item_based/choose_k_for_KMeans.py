import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils import choose_k_for_KMeans

k_for_test = 80
train_size = 1000000
train_data = pd.read_csv("../dmt-2025-2nd-assignment/training_set_VU_DM.csv", nrows=train_size)

# # for K-Means by users
# features = [
#     'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
#     'srch_children_count', 'srch_room_count', 'orig_destination_distance'
# ]

# # for K-Means by search
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

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

train_X = train_data[features]
train_X = imputer.fit_transform(train_X)
train_X = scaler.fit_transform(train_X)

choose_k_for_KMeans(train_X, k_for_test)