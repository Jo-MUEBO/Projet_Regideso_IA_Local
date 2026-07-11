import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib # Pour enregistrer le "cerveau" de l'IA une fois entraîné

# Chargement du dataset
df = pd.read_csv('consommation_regideso.csv')

# On ne garde que les colonnes numériques pour l'IA
# L'IA ne comprend pas les mots comme "Lemba", elle veut des chiffres
features = ['consommation_m3', 'pression_bar', 'temperature_c']
X = df[features]

# Configuration du modèle
# contamination=0.01 signifie qu'on s'attend à environ 1% d'anomalies
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# Entraînement : L'IA observe les 48 000 lignes pour comprendre la "normale"
print("L'IA analyse les données de consommation de Kinshasa...")
model.fit(X)

# L'IA prédit : 1 pour normal, -1 pour anomalie
df['prediction_ia'] = model.predict(X)

# On transforme le -1 en 1 pour que ce soit plus clair (1 = Fuite détectée)
df['ia_alerte'] = df['prediction_ia'].apply(lambda x: 1 if x == -1 else 0)

# Sauvegarder le modèle (le cerveau) dans un fichier
joblib.dump(model, 'modele_regideso.pkl')

# Sauvegarder les données avec les prédictions
df.to_csv('resultats_detection.csv', index=False)

print("Analyse terminée ! Le modèle est prêt et les résultats sont dans 'resultats_detection.csv'.")