import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Laden der Daten
heart = pd.read_csv("C:/Users/Scele/OneDrive/Desktop/Modul12/heartdata.csv", index_col=0)

# Verteilung
print(heart[['biking', 'smoking']].corr())
sns.pairplot(heart[['heartdisease', 'smoking', 'biking']])
plt.show()

# Lineare Regression
X = heart[['biking', 'smoking']]
y = heart['heartdisease']
lm = LinearRegression()
lm.fit(X, y)
print("R²:", lm.score(X, y))

# Plotten um das Modell zu prüfen
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(lm.predict(X), y - lm.predict(X))
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')

plt.subplot(2, 2, 2)
plt.scatter(lm.predict(X), y)
plt.title('Fitted vs Actual')
plt.xlabel('Fitted values')
plt.ylabel('Actual values')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

# Plotten
plt.subplot(2, 2, 3)
sns.regplot(x='heartdisease', y='biking', data=heart, color='green')
plt.subplot(2, 2, 4)
sns.regplot(x='heartdisease', y='smoking', data=heart, color='red')
plt.show()

# Modelltraining
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lm_cv = make_pipeline(StandardScaler(), LinearRegression())
scores = cross_val_score(lm_cv, X, y, cv=10)
print("Mean R² (Cross-Validation):", np.mean(scores))
