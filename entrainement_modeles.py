# ==============================================================================
# ENTRAINEMENT_MODELES.PY - ENTRAINEMENT DES MODELES ISOLATION FOREST / OC-SVM
# Ce script se lance UNE SEULE FOIS (ou a chaque fois qu'on veut reentrainer),
# a partir du fichier historique genere par generateur.py. Il sauvegarde les
# modeles entraines sur disque (.pkl) pour que moteur_ia.py puisse ensuite les
# charger et les utiliser en continu, sans jamais avoir a les reentrainer.
# ==============================================================================
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib

# --- CONFIGURATION ---
FICHIER_DONNEES = "consommation_regideso.csv"
FICHIER_SCALER = "regideso_scaler.pkl"
FICHIER_MODELE_IFOREST = "regideso_modele_iforest.pkl"
FICHIER_MODELE_OCSVM = "regideso_modele_ocsvm.pkl"

# Colonnes utilisees comme variables d'entree des modeles. On reste volontairement
# sur pression_bar et consommation_m3 : ce sont les deux variables dont la
# correlation inverse anormale caracterise une fuite (voir Chapitre II de la
# monographie : chute de pression + debit soutenu = signature d'une fuite).
COLONNES_FEATURES = ["pression_bar", "consommation_m3"]


def entrainer():
    print("Chargement des données historiques...")
    df = pd.read_csv(FICHIER_DONNEES)

    # Proportion reelle de fuites dans le jeu de donnees etiquete : utilisee
    # pour calibrer le taux de contamination attendu par Isolation Forest.
    taux_contamination = df["est_une_fuite_reelle"].mean()
    print(f"Taux de fuites observé dans les données : {taux_contamination:.4f}")

    X = df[COLONNES_FEATURES].values

    # --- Normalisation ---
    # Indispensable pour One-Class SVM (base sur les distances), et bonne
    # pratique egalement pour Isolation Forest.
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)

    # --- Isolation Forest ---
    print("Entraînement d'Isolation Forest...")
    modele_iforest = IsolationForest(
        n_estimators=150,
        contamination=max(taux_contamination, 0.001),   # jamais 0, sinon erreur sklearn
        random_state=42,
    )
    modele_iforest.fit(X_scale)

    # --- One-Class SVM ---
    print("Entraînement de One-Class SVM...")
    modele_ocsvm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=max(taux_contamination, 0.001),
    )
    modele_ocsvm.fit(X_scale)

    # --- Sauvegarde sur disque ---
    joblib.dump(scaler, FICHIER_SCALER)
    joblib.dump(modele_iforest, FICHIER_MODELE_IFOREST)
    joblib.dump(modele_ocsvm, FICHIER_MODELE_OCSVM)

    print("Modèles entraînés et sauvegardés :")
    print(f"  - {FICHIER_SCALER}")
    print(f"  - {FICHIER_MODELE_IFOREST}")
    print(f"  - {FICHIER_MODELE_OCSVM}")


if __name__ == "__main__":
    entrainer()