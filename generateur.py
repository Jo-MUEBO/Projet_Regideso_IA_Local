import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. CONFIGURATION
np.random.seed(42) # Pour que les résultats soient identiques à chaque test
communes_kin = [
    "Bandalungwa", "Barumbu", "Bumbu", "Gombe", "Kalamu", "Kasa-Vubu", 
    "Kisenso", "Kinshasa", "Kintambo", "Lemba", "Limete", "Lingwala", 
    "Makala", "Maluku", "Masina", "Matete", "Mont-Ngafula", "Ndjili", 
    "Ngaba", "Ngaliema", "Ngiri-Ngiri", "Nsele", "Olu", "Selembao"
]

start_date = datetime(2026, 1, 1, 0, 0)
heures_par_commune = 2000 # Environ 3 mois de données par commune
data_list = []

# 2. FONCTION DE LOGIQUE MÉTIER (Consommation d'eau)
def obtenir_conso(heure, commune):
    # Logique de base selon l'heure
    if 6 <= hour <= 9:      # Matin
        val = np.random.uniform(45, 65)
    elif 18 <= hour <= 21:  # Soir
        val = np.random.uniform(50, 75)
    elif 0 <= hour <= 4:    # Nuit
        val = np.random.uniform(2, 7)
    else:                   # Journée normale
        val = np.random.uniform(15, 35)
    
    # Ajustement par type de commune (Scalabilité)
    if commune == "Gombe": # Plus de bureaux/industries
        val *= 1.3
    if commune in ["Maluku", "Nsele"]: # Zones plus rurales/vastes
        val *= 0.8
        
    return val

# 3. GÉNÉRATION DES DONNÉES
print("Génération des données pour les 24 communes...")

for commune in communes_kin:
    for i in range(heures_par_commune):
        current_time = start_date + timedelta(hours=i)
        hour = current_time.hour
        
        conso = obtenir_conso(hour, commune)
        pression = np.random.uniform(3.5, 4.5) # Pression normale en Bar
        temp = np.random.uniform(26, 33)       # Température Kinshasa
        
        data_list.append({
            'timestamp': current_time,
            'commune': commune,
            'consommation_m3': round(conso, 2),
            'pression_bar': round(pression, 2),
            'temperature_c': round(temp, 1),
            'est_une_fuite_reelle': 0 # On marquera les fuites après
        })

# Transformation en tableau (DataFrame)
df = pd.DataFrame(data_list)

# 4. INJECTION DES ANOMALIES (Les fuites)
# On crée 300 fuites aléatoires sur les 48 000 lignes totales
print("Injection des fuites de la REGIDESO...")
indices_fuites = np.random.choice(df.index, size=300, replace=False)

for idx in indices_fuites:
    # Simulation d'une fuite : La consommation explose et la pression chute
    df.loc[idx, 'consommation_m3'] = round(df.loc[idx, 'consommation_m3'] * np.random.uniform(2.5, 5.0), 2)
    df.loc[idx, 'pression_bar'] = round(np.random.uniform(0.3, 1.2), 2)
    df.loc[idx, 'est_une_fuite_reelle'] = 1

# 5. SAUVEGARDE EN CSV
df.to_csv('consommation_regideso.csv', index=False)
print(f"Terminé ! Fichier créé avec {len(df)} lignes.")