import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Laden der Feather-Datei
cox2_data = pd.read_feather("C:/Users/Scele/OneDrive/Desktop/Modul12/cox2_data.feather")

# Features und Zielvariable trennen
X = cox2_data.drop(columns=['class'])
y = cox2_data['class']

# Umwandlung der Zielvariable in numerische Werte
le = LabelEncoder()
y = le.fit_transform(y)

# Nun können Sie die Zielvariable in Integer konvertieren
y = y.astype(int)


print(cox2_data.head())

# Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Modelltraining und -evaluation mit Random Forest
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123))
rf_pipeline.fit(X_train, y_train)

# Vorhersagen auf Testdaten
y_pred = rf_pipeline.predict(X_test)

# Konfusionsmatrix und Klassifikationsbericht ausgeben
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Kreuzvalidierung
cross_val_scores = cross_val_score(rf_pipeline, X, y, cv=10)
print("Mean Accuracy (Cross-Validation):", cross_val_scores.mean())
