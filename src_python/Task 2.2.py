import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2

# Laden der Datei
diab = pd.read_csv("C:/Users/Scele/OneDrive/Desktop/Modul12/diabetes.csv")
print(diab.head())
print(diab.shape)
print(diab.isna().sum())

# Outlier-Ersatz basierend auf IQR
def outlier_detection(x):
    Q1 = np.nanquantile(x, 0.25)
    Q3 = np.nanquantile(x, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    x[(x < lower_bound) | (x > upper_bound)] = np.nan
    return x

numeric_cols = diab.select_dtypes(include=np.number).columns
diab[numeric_cols] = diab[numeric_cols].apply(outlier_detection)

# 0 ausschließen
zero_cols = ['plas', 'pres', 'skin', 'insu', 'mass']
diab[zero_cols] = diab[zero_cols].replace(0, np.nan)

# NA-Werte entfernen
diab_cleaned = diab.dropna()

# Daten beschreiben
print(diab_cleaned.describe())

# Informationen gewinnen
X = diab_cleaned.drop(columns=['class'])
y = diab_cleaned['class']

ig = mutual_info_classif(X, y)
ig_df = pd.DataFrame({"Feature": X.columns, "Information_Gain": ig})
ig_sorted = ig_df.sort_values(by="Information_Gain", ascending=False)
print(ig_sorted)

# Visualisierung
highest_ig_feature = ig_sorted.iloc[0]["Feature"]
smallest_ig_feature = ig_sorted.iloc[-1]["Feature"]

plt.figure(figsize=(12, 6))
sns.boxplot(x=highest_ig_feature, data=diab_cleaned)
plt.title(f"Boxplot of {highest_ig_feature}")
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(x=highest_ig_feature, data=diab_cleaned, fill=True)
plt.title(f"Distribution plot of {highest_ig_feature}")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=smallest_ig_feature, data=diab_cleaned)
plt.title(f"Boxplot of {smallest_ig_feature}")
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(x=smallest_ig_feature, data=diab_cleaned, fill=True)
plt.title(f"Distribution plot of {smallest_ig_feature}")
plt.show()
