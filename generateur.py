# ==============================================================================
# GENERATEUR.PY - GENERATION DES DONNEES HISTORIQUES POUR L'ENTRAINEMENT
# Cree un jeu de donnees simule avec des fuites CONNUES (etiquetees), utilise
# ensuite par entrainement_modeles.py pour entrainer Isolation Forest et
# One-Class SVM. Ce script tourne UNE SEULE FOIS (pas en continu).
# ==============================================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
np.random.seed(42)   # Pour que les resultats soient identiques a chaque test

communes_kin = [
    "Bandalungwa", "Barumbu", "Bumbu", "Gombe", "Kalamu", "Kasa-Vubu",
    "Kisenso", "Kinshasa", "Kintambo", "Lemba", "Limete", "Lingwala",
    "Makala", "Maluku", "Masina", "Matete", "Mont-Ngafula", "Ndjili",
    "Ngaba", "Ngaliema", "Ngiri-Ngiri", "Nsele", "Selembao"
]

start_date = datetime(2026, 1, 1, 0, 0)
heures_par_commune = 2000   # Environ 3 mois de donnees par commune
data_list = []


# --- 2. FONCTION DE LOGIQUE METIER (Consommation d'eau) ---
def obtenir_conso(heure, commune):
    """
    Simule une consommation realiste selon l'heure de la journee et le
    profil de la commune. CORRECTION : le parametre s'appelle bien 'heure'
    et est utilise a l'interieur de la fonction (au lieu de dependre d'une
    variable globale 'hour' comme dans la version precedente, ce qui rendait
    la fonction fragile et incorrecte si elle etait appelee ailleurs que
    juste apres la boucle principale).
    """
    # Logique de base selon l'heure
    if 6 <= heure <= 9:        # Matin
        val = np.random.uniform(45, 65)
    elif 18 <= heure <= 21:    # Soir
        val = np.random.uniform(50, 75)
    elif 0 <= heure <= 4:      # Nuit
        val = np.random.uniform(2, 7)
    else:                      # Journee normale
        val = np.random.uniform(15, 35)

    # Ajustement par type de commune (scalabilite)
    if commune == "Gombe":                      # Plus de bureaux/industries
        val *= 1.3
    if commune in ["Maluku", "Nsele"]:           # Zones plus rurales/vastes
        val *= 0.8

    return val


# --- 3. GENERATION DES DONNEES ---
print("Génération des données pour les communes...")

for commune in communes_kin:
    for i in range(heures_par_commune):
        current_time = start_date + timedelta(hours=i)
        hour = current_time.hour

        conso = obtenir_conso(hour, commune)     # 'hour' passe bien en parametre 'heure'
        pression = np.random.uniform(3.5, 4.5)   # Pression normale en Bar
        temp = np.random.uniform(26, 33)         # Temperature Kinshasa

        data_list.append({
            'timestamp': current_time,
            'commune': commune,
            'consommation_m3': round(conso, 2),
            'pression_bar': round(pression, 2),
            'temperature_c': round(temp, 1),
            'est_une_fuite_reelle': 0   # Marque ci-dessous pour les fuites injectees
        })

df = pd.DataFrame(data_list)

# --- 4. INJECTION DES ANOMALIES (les fuites connues, utilisees pour l'entrainement) ---
print("Injection des fuites simulées de la REGIDESO...")
nb_fuites = 300
indices_fuites = np.random.choice(df.index, size=nb_fuites, replace=False)

for idx in indices_fuites:
    # Simulation d'une fuite : la consommation explose et la pression chute
    df.loc[idx, 'consommation_m3'] = round(df.loc[idx, 'consommation_m3'] * np.random.uniform(2.5, 5.0), 2)
    df.loc[idx, 'pression_bar'] = round(np.random.uniform(0.3, 1.2), 2)
    df.loc[idx, 'est_une_fuite_reelle'] = 1

# --- 5. SAUVEGARDE EN CSV ---
NOM_FICHIER_SORTIE = 'consommation_regideso.csv'
df.to_csv(NOM_FICHIER_SORTIE, index=False)
print(f"Terminé ! Fichier '{NOM_FICHIER_SORTIE}' créé avec {len(df)} lignes "
      f"({nb_fuites} fuites étiquetées, {len(df) - nb_fuites} points normaux).")