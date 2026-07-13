# ==============================================================================
# GENERATEUR_STREAMING.PY - INJECTION CONTINUE DE DONNEES (SIMULATION TEMPS REEL)
# Simule l'arrivee continue de mesures des capteurs, toutes les 4 secondes,
# et les insere dans la base SQLite lue par app.py.
#
# CORRECTION IMPORTANTE : le nom de la base de donnees DOIT etre identique a
# celui utilise dans app.py (DB_NAME = "regideso_comparatif.db"). Une
# incoherence de nom entre les deux scripts est la cause la plus frequente
# d'un tableau de bord qui reste bloque sur "En attente du flux..." alors
# que le generateur tourne bel et bien.
# ==============================================================================
import sqlite3
import time
import random
import numpy as np
from datetime import datetime

# --- CONFIGURATION LOCALE ---
# IMPORTANT : ce nom doit etre EXACTEMENT le meme que DB_NAME dans app.py
DB_NAME = "regideso_comparatif.db"

# Cadence d'insertion en secondes. Doit rester coherente avec le rythme de
# rafraichissement de app.py (st.fragment run_every=4) pour un flux fluide.
INTERVALLE_SECONDES = 4

# Probabilite qu'une fuite reelle soit simulee sur une commune a un cycle
# donne. Une valeur faible (proche du taux reel utilise a l'entrainement,
# voir entrainement_modeles.py) evite de noyer le systeme d'alertes en
# permanence, tout en garantissant que des fuites apparaissent regulierement
# pour pouvoir demontrer le systeme.
PROBABILITE_FUITE = 0.01   # ~1% de chance par commune et par cycle

COMMUNES = [
    "Bandalungwa", "Barumbu", "Bumbu", "Gombe", "Kalamu", "Kasa-Vubu",
    "Kisenso", "Kinshasa", "Kintambo", "Lemba", "Limete", "Lingwala",
    "Makala", "Maluku", "Masina", "Matete", "Mont-Ngafula", "Ndjili",
    "Ngaba", "Ngaliema", "Ngiri-Ngiri", "Nsele", "Selembao",
]


def obtenir_conso_normale(heure, commune):
    """
    Reproduit EXACTEMENT la meme logique de consommation "normale" que
    generateur.py (voir ce fichier). Il est essentiel que le comportement
    normal simule ici corresponde a celui vu par les modeles a l'entrainement,
    sans quoi les modeles interpretent le flux live comme anormal en
    permanence, meme en l'absence de toute fuite reelle.
    """
    if 6 <= heure <= 9:
        val = np.random.uniform(45, 65)
    elif 18 <= heure <= 21:
        val = np.random.uniform(50, 75)
    elif 0 <= heure <= 4:
        val = np.random.uniform(2, 7)
    else:
        val = np.random.uniform(15, 35)

    if commune == "Gombe":
        val *= 1.3
    if commune in ["Maluku", "Nsele"]:
        val *= 0.8

    return val


def init_db():
    """Cree les tables necessaires si elles n'existent pas deja."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS mesures_comparatives
                      (timestamp TEXT, commune TEXT, consommation_m3 FLOAT,
                       pression_bar FLOAT, temperature_c FLOAT,
                       alerte_iforest SMALLINT, alerte_ocsvm SMALLINT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS topologie_secteurs
                      (timestamp TEXT, commune TEXT, debit_source FLOAT,
                       debit_secteur_a FLOAT, debit_secteur_b FLOAT, perte_calculee FLOAT)''')
    conn.commit()
    conn.close()


def injecter_donnees():
    """
    Insere une nouvelle mesure par commune, avec un comportement "normal"
    coherent avec les donnees d'entrainement (voir obtenir_conso_normale),
    et une fuite reelle simulee occasionnellement (PROBABILITE_FUITE) pour
    pouvoir demontrer le systeme de detection.

    Les colonnes alerte_iforest et alerte_ocsvm sont inserees a NULL (et non
    0) : NULL signifie "pas encore analysee par le moteur IA". C'est
    moteur_ia.py qui remplit ensuite ces colonnes avec le vrai resultat de
    la detection (0 = normal, 1 = anomalie).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    timestamp_dt = datetime.now()
    timestamp = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
    heure_actuelle = timestamp_dt.hour

    for commune in COMMUNES:
        consommation = round(obtenir_conso_normale(heure_actuelle, commune), 2)
        pression = round(random.uniform(3.5, 4.5), 2)   # meme plage "normale" qu'a l'entrainement

        # Simulation occasionnelle d'une vraie fuite, avec la meme signature
        # que celle utilisee a l'entrainement (voir generateur.py) : la
        # consommation explose et la pression chute brutalement.
        if random.random() < PROBABILITE_FUITE:
            consommation = round(consommation * random.uniform(2.5, 5.0), 2)
            pression = round(random.uniform(0.3, 1.2), 2)
            print(f"  \u26a0 Fuite simulée injectée sur {commune} !")

        # CORRECTION ICI : On précise les 7 colonnes cibles
        cursor.execute(
            "INSERT INTO mesures_comparatives (timestamp, commune, consommation_m3, pression_bar, temperature_c, alerte_iforest, alerte_ocsvm) VALUES (?,?,?,?,?,?,?)",
            (timestamp, commune, consommation, pression, 25.0, None, None)
        )

        # CORRECTION ICI : On précise les 6 colonnes cibles
        cursor.execute(
            "INSERT INTO topologie_secteurs (timestamp, commune, debit_source, debit_secteur_a, debit_secteur_b, perte_calculee) VALUES (?,?,?,?,?,?)",
            (timestamp, commune, 100.0, 40.0, 40.0, 20.0)
        )

    conn.commit()
    conn.close()
    print(f"[{timestamp}] {len(COMMUNES)} communes injectées dans {DB_NAME}")

if __name__ == "__main__":
    init_db()
    print(f"Système de simulation activé (base : {DB_NAME}, "
          f"cadence : {INTERVALLE_SECONDES}s) ...")
    while True:
        injecter_donnees()
        time.sleep(INTERVALLE_SECONDES)