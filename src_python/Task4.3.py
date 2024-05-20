import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

# Annahme: Die BloodBrain-Daten sind bereits in CSV-Dateien gespeichert.
# Wenn nicht, müssen sie zuerst in CSV-Dateien konvertiert werden.
bbbDescr = pd.read_csv("C:/Users/Scele/OneDrive/Desktop/Modul12/bbbDescr.csv")
logBBB = pd.read_csv("C:/Users/Scele/OneDrive/Desktop/Modul12/logBBB.csv")

# Daten vorbereiten
x = bbbDescr
y = logBBB.squeeze()  # Konvertiert DataFrame in Series

# Training- und Testdaten aufteilen
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Feature-Varianz überprüfen
featVar = x_train.var(axis=0)
print(f"Anzahl der Features mit Varianz < 0.001: {sum(featVar < 0.001)}")

# Modelltraining und -bewertung mit Random Forest
trControl = {'cv': 10}
pipeline = make_pipeline(StandardScaler(), RandomForestRegressor())
pipeline.fit(x_train, y_train)

# Feature-Importanz berechnen
importances = pipeline.named_steps['randomforestregressor'].feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importance:")
for f in range(x_train.shape[1]):
    print(f"{x_train.columns[indices[f]]}: {importances[indices[f]]}")

# Modellvorhersage und RMSE-Berechnung
y_predicted = pipeline.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
print(f"RMSE: {rmse}")
