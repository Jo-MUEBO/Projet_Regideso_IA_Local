# ==============================================================================
# MOTEUR_IA.PY - LE "MIDDLEWARE" MANQUANT
# Tourne en parallele de generateur_streaming.py et de app.py. Son role :
# lire les lignes fraichement inserees (alerte_iforest / alerte_ocsvm = NULL),
# leur appliquer les modeles Isolation Forest et One-Class SVM deja entraines
# (voir entrainement_modeles.py), et mettre a jour la base avec le vrai
# resultat de la detection. Sans ce script, app.py affichera toujours un
# reseau "stable" puisque personne ne calcule jamais les alertes.
# ==============================================================================
import sqlite3
import time
import joblib
import numpy as np

# --- CONFIGURATION ---
# IMPORTANT : meme nom de base que app.py et generateur_streaming.py
DB_NAME = "regideso_comparatif.db"

FICHIER_SCALER = "regideso_scaler.pkl"
FICHIER_MODELE_IFOREST = "regideso_modele_iforest.pkl"
FICHIER_MODELE_OCSVM = "regideso_modele_ocsvm.pkl"

# Cadence d'analyse : toutes les X secondes, on va chercher les nouvelles
# lignes non encore analysees et on les score. Une cadence plus courte que
# celle du generateur (4s) garantit qu'aucune ligne ne reste NULL longtemps.
INTERVALLE_SECONDES = 2

# Nombre maximal de lignes traitees par cycle (evite de tout recharger d'un
# coup si le moteur a ete arrete un moment et qu'un retard s'est accumule)
LOT_MAX_PAR_CYCLE = 500


def charger_modeles():
    """Charge le scaler et les deux modeles entraines depuis le disque."""
    scaler = joblib.load(FICHIER_SCALER)
    modele_iforest = joblib.load(FICHIER_MODELE_IFOREST)
    modele_ocsvm = joblib.load(FICHIER_MODELE_OCSVM)
    return scaler, modele_iforest, modele_ocsvm


def recuperer_lignes_non_analysees(conn):
    """
    Recupere les lignes dont l'alerte n'a pas encore ete calculee
    (alerte_iforest IS NULL), avec leur rowid SQLite pour pouvoir les
    mettre a jour individuellement ensuite.
    """
    cursor = conn.cursor()
    cursor.execute(
        """SELECT rowid, pression_bar, consommation_m3
           FROM mesures_comparatives
           WHERE alerte_iforest IS NULL
           LIMIT ?""",
        (LOT_MAX_PAR_CYCLE,)
    )
    return cursor.fetchall()


def calculer_alertes(lignes, scaler, modele_iforest, modele_ocsvm):
    """
    Calcule les predictions des deux modeles pour un lot de lignes.
    Convention : 1 = anomalie (fuite suspectee), 0 = comportement normal.
    scikit-learn renvoie -1 (anomalie) / 1 (normal) pour ces deux modeles :
    on convertit donc vers la convention 1/0 utilisee dans app.py.
    """
    rowids = [ligne[0] for ligne in lignes]
    X = np.array([[ligne[1], ligne[2]] for ligne in lignes])   # pression_bar, consommation_m3
    X_scale = scaler.transform(X)

    pred_iforest = modele_iforest.predict(X_scale)   # -1 ou 1
    pred_ocsvm = modele_ocsvm.predict(X_scale)        # -1 ou 1

    alerte_iforest = [1 if p == -1 else 0 for p in pred_iforest]
    alerte_ocsvm = [1 if p == -1 else 0 for p in pred_ocsvm]

    return rowids, alerte_iforest, alerte_ocsvm


def mettre_a_jour_alertes(conn, rowids, alerte_iforest, alerte_ocsvm):
    """Met a jour chaque ligne individuellement via son rowid SQLite."""
    cursor = conn.cursor()
    for rowid, a_if, a_oc in zip(rowids, alerte_iforest, alerte_ocsvm):
        cursor.execute(
            """UPDATE mesures_comparatives
               SET alerte_iforest = ?, alerte_ocsvm = ?
               WHERE rowid = ?""",
            (a_if, a_oc, rowid)
        )
    conn.commit()


def boucle_principale():
    print("Chargement des modèles entraînés...")
    scaler, modele_iforest, modele_ocsvm = charger_modeles()
    print("Moteur IA activé — analyse en continu des nouvelles mesures...")

    while True:
        conn = sqlite3.connect(DB_NAME)
        try:
            lignes = recuperer_lignes_non_analysees(conn)
            if lignes:
                rowids, alerte_iforest, alerte_ocsvm = calculer_alertes(
                    lignes, scaler, modele_iforest, modele_ocsvm
                )
                mettre_a_jour_alertes(conn, rowids, alerte_iforest, alerte_ocsvm)
                nb_anomalies = sum(alerte_iforest)
                print(f"{len(lignes)} ligne(s) analysée(s), "
                      f"{nb_anomalies} anomalie(s) détectée(s) (Isolation Forest).")
            else:
                print("Aucune nouvelle ligne à analyser.")
        except sqlite3.OperationalError as e:
            # La base peut ne pas exister au tout premier lancement si
            # generateur_streaming.py n'a pas encore ete demarre : on
            # attend simplement le prochain cycle plutot que de planter.
            print(f"Base de données pas encore prête ({e}), nouvelle tentative...")
        finally:
            conn.close()

        time.sleep(INTERVALLE_SECONDES)


if __name__ == "__main__":
    boucle_principale()