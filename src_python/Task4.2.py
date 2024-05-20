import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff

# Laden der Daten
data, meta = arff.loadarff("C:/Users/Scele/OneDrive/Desktop/Modul12/diabetes.arff")
diab = pd.DataFrame(data)

# Umwandlung der Zielvariable in numerische Werte
le = LabelEncoder()
diab['class'] = le.fit_transform(diab['class'])

# GLM (generalized linear model)
X = diab.drop(columns=['class'])
y = diab['class']
glm = LogisticRegression(max_iter=1000)
glm.fit(X, y)
print("Summary:")
print(glm.coef_)
print(glm.intercept_)

# Modell Training mit Kreuzvalidierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
glm_cv = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(glm_cv, X, y, cv=10)
print("Mean Accuracy (Cross-Validation):", scores.mean())
