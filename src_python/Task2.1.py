import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Daten
dsta = pd.read_csv("C:/Users/Scele/OneDrive/Desktop/Modul12/food.csv", index_col=0)
print(dsta.shape)
print(dsta.isna().sum())

# PCA (Principal Component Analysis)
scaler = StandardScaler()
scaled_dsta = scaler.fit_transform(dsta.iloc[:, 1:])

pca = PCA()
pca.fit(scaled_dsta)

# Summary
print(pca.explained_variance_ratio_)

# PCA (Principal Component Analysis) plot
scores = pd.DataFrame(pca.transform(scaled_dsta), columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)])

plt.figure(figsize=(8, 6))
plt.scatter(scores["PC1"], scores["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Score Plot")
plt.grid(True)
plt.show()