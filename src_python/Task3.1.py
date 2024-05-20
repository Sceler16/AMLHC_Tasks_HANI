import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score


# Laden der Daten
dsta = pd.read_csv("C:/Users/Scele/OneDrive/Desktop/Modul12/food.csv", index_col=0)
fds = StandardScaler().fit_transform(dsta)

# Clusterung
best_k = 0
best_silu = -np.inf
for k in range(2, 6):
    kmc = KMeans(n_clusters=k).fit(fds)
    silhouette_value = silhouette_score(fds, kmc.labels_)
    print("Silhouette coefficient for k =", k, ":", silhouette_value)
    if silhouette_value > best_silu:
        best_k = k
        best_silu = silhouette_value

print("\nFinal selected number of clusters:", best_k)

# PCA
pca = PCA(n_components=2)
data_red = pca.fit_transform(fds)
data_red *= -1
plt.scatter(data_red[:, 0], data_red[:, 1], c=kmc.labels_)
for i, txt in enumerate(dsta.index):
    plt.text(data_red[i, 0], data_red[i, 1] - 0.1, txt, fontsize=8)

plt.show()
print(pca.explained_variance_ratio_)

# Hirarchical clustering
hcf = dendrogram(linkage(fds, method='ward'))

# DBC density based clustering
dbc = DBSCAN(eps=3).fit(fds)
clus = dbc.labels_
print(clus)
plt.scatter(data_red[:, 0], data_red[:, 1], c=clus)
for i, txt in enumerate(dsta.index):
    plt.text(data_red[i, 0], data_red[i, 1], txt, fontsize=8)

plt.show()
# Heatmap
plt.figure(figsize=(10, 7))
plt.imshow(fds, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Heatmap of Data')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.show()