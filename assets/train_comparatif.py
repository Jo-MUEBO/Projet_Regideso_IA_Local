import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib

print(" Génération des données d'entraînement...")
# Simulation de 1000 données normales (débit stable, bonne pression)
normal_conso = np.random.uniform(35.0, 85.0, 950)
normal_pression = np.random.uniform(2.8, 4.2, 950)
normal_temp = np.random.uniform(24.0, 29.0, 950)

# Simulation de 50 anomalies (fuites : gros débit, chute de pression)
anomalie_conso = np.random.uniform(160.0, 240.0, 50)
anomalie_pression = np.random.uniform(0.2, 0.7, 50)
anomalie_temp = np.random.uniform(24.0, 29.0, 50)

# Assemblage du dataset
df_train = pd.DataFrame({
    'consommation_m3': np.concatenate([normal_conso, anomalie_conso]),
    'pression_bar': np.concatenate([normal_pression, anomalie_pression]),
    'temperature_c': np.concatenate([normal_temp, anomalie_temp])
})

# 1. Entraînement de l'Isolation Forest
print(" Entraînement d'Isolation Forest...")
model_iforest = IsolationForest(contamination=0.05, random_state=42)
model_iforest.fit(df_train)
joblib.dump(model_iforest, 'modele_iforest.pkl')

# 2. Entraînement du One-Class SVM
print(" Entraînement de One-Class SVM...")
model_ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
model_ocsvm.fit(df_train)
joblib.dump(model_ocsvm, 'modele_ocsvm.pkl')

print("💾 Les deux modèles ont été entraînés et sauvegardés avec succès !")