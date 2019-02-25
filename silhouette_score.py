from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

dataset = pd.read_csv('College.csv')

dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset.drop(['Private'], axis=1)

wcss=[]

# Elbow method to know the number of clusters

for i in range(2, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    x = kmeans.fit_predict(dataset)
    wcss.append(kmeans.inertia_)

    sil = silhouette_score(dataset, x)
    print("Cluster # ", i, ": Silhouette score = ", sil)

plt.plot(range(2, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
plt.show()
