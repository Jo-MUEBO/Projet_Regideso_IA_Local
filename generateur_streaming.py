import time
import random
import pandas as pd
import joblib
from datetime import datetime
from supabase import create_client, Client

# =====================================================================
# 1. CONFIGURATION SUPABASE (Remplace par tes vraies informations !)
# =====================================================================
SUPABASE_URL = "https://tglrpwuvyjjmiqzuxdmb.supabase.co/rest/v1/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRnbHJwd3V2eWpqbWlxenV4ZG1iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODM3MTcxODUsImV4cCI6MjA5OTI5MzE4NX0.RsP7b1jsunNohkCERq2KZ2wYxyNyfhEF1C4rW_ZIIqE"


print("🔄 Connexion au serveur cloud Supabase en cours...")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connexion à Supabase établie avec succès !")
except Exception as e:
    print(f"❌ Erreur de connexion Supabase : {e}")
    exit()

# =====================================================================
# 2. CHARGEMENT DES MODÈLES D'INTELLIGENCE ARTIFICIELLE
# =====================================================================
try:
    model_iforest = joblib.load('modele_iforest.pkl')
    model_ocsvm = joblib.load('modele_ocsvm.pkl')
    print("🧠 Modèles IA chargés (Isolation Forest & One-Class SVM).")
except FileNotFoundError:
    print("❌ Erreur : Modèles manquants. Lancez d'abord 'train_comparatif.py'.")
    exit()

COMMUNES = [
    "Bandalungwa", "Barumbu", "Bumbu", "Gombe", "Kalamu", "Kasa-Vubu", 
    "Kimbanseke", "Kinshasa", "Kintambo", "Kisenso", "Lemba", "Limete", 
    "Lingwala", "Makala", "Maluku", "Masina", "Matete", "Mont Ngafula", 
    "Ndjili", "Ngaba", "Ngaliema", "Ngiri-Ngiri", "Nsele", "Selembao"
]

print("📡 Flux IoT topologique activé vers le Cloud Supabase...")

# =====================================================================
# 3. BOUCLE DE GÉNÉRATION ET D'ENVOI EN TEMPS RÉEL
# =====================================================================
try:
    while True:
        maintenant = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # On va stocker les insertions par lots pour optimiser la vitesse réseau
        lot_mesures = []
        lot_topologie = []
        
        for commune in COMMUNES:
            # --- COMPORTEMENT GLOBAL ---
            est_une_fuite = random.random() < 0.06
            if est_une_fuite:
                conso = random.uniform(150.0, 230.0)
                pression = random.uniform(0.3, 0.8)
            else:
                conso = random.uniform(40.0, 80.0)
                pression = random.uniform(2.5, 4.0)
            temp = random.uniform(25.0, 28.0)
            
            # --- PRÉDICTION DES MODÈLES IA ---
            donnee_live = pd.DataFrame({'consommation_m3': [conso], 'pression_bar': [pression], 'temperature_c': [temp]})
            pred_iforest = 1 if model_iforest.predict(donnee_live)[0] == -1 else 0
            pred_ocsvm = 1 if model_ocsvm.predict(donnee_live)[0] == -1 else 0
            
            # Préparation de la ligne pour `mesures_comparatives`
            lot_mesures.append({
                "timestamp": maintenant,
                "commune": commune,
                "consommation_m3": round(conso, 2),
                "pression_bar": round(pression, 2),
                "temperature_c": round(temp, 2),
                "alerte_iforest": int(pred_iforest),
                "alerte_ocsvm": int(pred_ocsvm)
            })
            
            # --- SIMULATION TOPOLOGIQUE (SOUS-SECTEURS) ---
            debit_source = conso
            
            if est_une_fuite:
                debit_a = debit_source * random.uniform(0.3, 0.4)
                debit_b = debit_source * random.uniform(0.2, 0.3)
            else:
                debit_a = debit_source * 0.55
                debit_b = debit_source * 0.43
                
            perte = max(0.0, debit_source - (debit_a + debit_b))
            
            # Préparation de la ligne pour `topologie_secteurs`
            lot_topologie.append({
                "timestamp": maintenant,
                "commune": commune,
                "debit_source": round(debit_source, 2),
                "debit_secteur_a": round(debit_a, 2),
                "debit_secteur_b": round(debit_b, 2),
                "perte_calculee": round(perte, 2)
            })
            
        # =============================================================
        # ENVOI DANS LE CLOUD (Remplaçant du conn.commit de SQLite)
        # =============================================================
        try:
            supabase.table("mesures_comparatives").insert(lot_mesures).execute()
            supabase.table("topologie_secteurs").insert(lot_topologie).execute()
            print(f"📥 [ {datetime.now().strftime('%H:%M:%S')} ] 24 communes injectées dans le Cloud avec succès !")
        except Exception as e_insert:
            print(f"⚠️ Erreur lors de l'envoi vers Supabase : {e_insert}")
            
        # Pause de 4 secondes avant le prochain envoi
        time.sleep(4)

except KeyboardInterrupt:
    print("\n🛑 Flux arrêté par l'utilisateur.")
except Exception as e_global:
    print(f"\n❌ Erreur critique : {e_global}")