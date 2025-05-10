import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_result(row):
    if row['booking_bool'] == 1:
        return 5
    elif row['click_bool'] == 1:
        return 1
    else:
        return 0

def round_to_0_1_5(x):
    distances = [abs(x - 0), abs(x - 1), abs(x - 5)]
    idx = np.argmin(distances)
    return [0, 1, 5][idx]

def choose_k_for_KMeans(X_scaled, max_range): 
    '''generate a figure of k vs. wcss'''
    wcss = []
    for k in range(2, max_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.plot(range(2, max_range), wcss, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal k')
    plt.show()
