from sklearn.cluster import KMeans
import pandas as pd


# Load college dataset
dataset = pd.read_csv('College.csv')

X_data = dataset.iloc[:, 2:]

# Print Data columns from dataset
print(X_data.columns)

nclusters = 3

seed = 7

# K-Means Clustering
km = KMeans(n_clusters=nclusters, random_state = seed)

km.fit(X_data)

y_cluster_kmeans=km.predict(X_data)


print(y_cluster_kmeans)

# dataset.info()
