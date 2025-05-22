import pandas as pd

filename = "../dmt-2025-2nd-assignment/training_set_VU_DM.csv"

results = []
search_score = 5
click_score = 50
booking_score = 100

chunk_iter = pd.read_csv(
    filename,
    usecols=['srch_id', 'prop_id', 'click_bool', 'booking_bool'],
    chunksize=100_000
)

for chunk in chunk_iter:
    chunk['score'] = search_score
    chunk.loc[chunk['click_bool'] == 0, 'score'] = search_score
    chunk.loc[(chunk['click_bool'] == 1) & (chunk['booking_bool'] == 0), 'score'] = click_score
    chunk.loc[chunk['booking_bool'] == 1, 'score'] = booking_score

    results.append(chunk[['srch_id', 'prop_id', 'score']])

all_scores = pd.concat(results, ignore_index=True)
all_scores = all_scores.groupby(['srch_id', 'prop_id'], as_index=False)['score'].max()
all_scores.to_csv(f"results/user_hotel_scores_{search_score}_{click_score}_{booking_score}.csv", index=False)