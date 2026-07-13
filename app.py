# ==============================================================================
# REGIDESO - SYSTEME INTELLIGENT DE DETECTION DE FUITES
# Application Streamlit de supervision, detection IA et cartographie du reseau
# ==============================================================================

# --- IMPORTS STANDARDS ET SCIENTIFIQUES ---
import streamlit as st                     # Framework web pour l'interface
import pandas as pd                        # Manipulation des donnees tabulaires
import numpy as np                         # Calcul numerique / generation de bruit aleatoire
import plotly.express as px                # Graphiques rapides (histogrammes, barres)
import plotly.graph_objects as go          # Graphiques avances (courbes, ROC, prevision)
import sqlite3                             # Connexion a la base de donnees locale
import time                                # Gestion des pauses pour le rafraichissement auto
from datetime import datetime              # Horodatage des rapports PDF
from PIL import Image                      # Chargement du logo de l'entreprise
from fpdf import FPDF                      # Generation du rapport PDF
from prophet import Prophet                # Modele de prevision de serie temporelle
import logging                             # Reduction du bruit de log genere par Prophet/cmdstanpy

# Prophet (via cmdstanpy) est tres verbeux par defaut (logs INFO a chaque entrainement).
# On relève le niveau de log pour ne garder que les avertissements/erreurs reels.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# --- IMPORTS POUR LA CARTOGRAPHIE SIG ---
import hashlib                             # Generation d'une position stable par fuite (seed deterministe)
import folium                              # Moteur de cartographie interactive (points cliquables)
from streamlit_folium import st_folium     # Pont entre Folium et Streamlit

# osmnx permet de recuperer le VRAI reseau routier (avenues/rues) depuis OpenStreetMap
# et d'y "aimanter" un point approximatif. Import protege : si la librairie n'est pas
# installee ou si le reseau n'est pas joignable au demarrage, l'app bascule automatiquement
# sur un positionnement approximatif plutot que de planter.
try:
    import osmnx as ox
    OSMNX_DISPONIBLE = True
except ImportError:
    OSMNX_DISPONIBLE = False


# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="REGIDESO - Détection de Fuites",   # Titre affiche dans l'onglet du navigateur
    page_icon=":material/water_drop:",              # Icone de l'onglet
    layout="wide",                                  # Utilise toute la largeur de l'ecran
    initial_sidebar_state="expanded"                # Sidebar ouverte par defaut
)


# ==============================================================================
# FEUILLE DE STYLE CSS - THEME NAVY / AQUA REGIDESO
# ==============================================================================
st.markdown("""
    <style>
        /* --- THEME GLOBAL DE L'APPLICATION --- */
        .stApp { background-color: #EDF5F9; }                              /* Fond general bleu tres clair */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }      /* Marges internes du contenu */
        h1, h2, h3, h4, h5, h6 { color: #0A2535 !important; font-weight: 800; letter-spacing: -0.5px; }
        p, span, div { color: #0A2535; }                                   /* Couleur de texte par defaut */

        /* --- SIDEBAR : DEGRADE NAVY --- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B2030, #0D2A3F, #0A1E2E) !important;
            min-width: 260px !important;
            max-width: 260px !important;
            border-right: none !important;
        }

        /* Texte clair pour tous les elements de la sidebar (contraste sur fond sombre) */
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] svg {
            color: #C8EEF5 !important;
        }

        /* --- CHAMP DE RECHERCHE DE LA SIDEBAR --- */
        [data-testid="stSidebar"] [data-testid="stTextInput"] div div {
            background-color: rgba(0, 200, 230, 0.07) !important;
            border: 1px solid rgba(0, 200, 230, 0.2) !important;
            color: #000000 !important;
            border-radius: 6px;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] input {
            color: #C8EEF5 !important;
        }

        /* --- NAVIGATION PRINCIPALE (st.radio transforme en menu lateral) --- */
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: transparent !important;
            border-left: 3px solid transparent !important;
            padding: 10px 14px !important;
            border-radius: 0px 8px 8px 0px !important;
            margin-bottom: 4px !important;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
            background-color: rgba(0, 200, 230, 0.05) !important;
        }

        /* Style de l'onglet actuellement selectionne */
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: rgba(0, 200, 230, 0.13) !important;
            border-left: 3px solid #00C8E6 !important;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] p,
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Titres de sections injectes visuellement dans le menu de navigation */
        div.row-widget.stRadio > div[role="radiogroup"] > label:nth-child(1)::before {
            content: "PRINCIPAL"; display: block; color: #4A8FA8; font-size: 11px;
            margin-bottom: 10px; font-weight: 800; letter-spacing: 1px;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label:nth-child(4)::before {
            content: "INTELLIGENCE IA"; display: block; color: #4A8FA8; font-size: 11px;
            margin-top: 18px; margin-bottom: 10px; font-weight: 800; letter-spacing: 1px;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label:nth-child(7)::before {
            content: "FINANCE & LOGS"; display: block; color: #4A8FA8; font-size: 11px;
            margin-top: 18px; margin-bottom: 10px; font-weight: 800; letter-spacing: 1px;
        }

        /* --- CARTES KPI / METRIQUES --- */
        [data-testid="stMetric"], .kpi-card {
            background-color: rgba(255, 255, 255, 0.80) !important;
            border: 1px solid rgba(0, 160, 190, 0.14) !important;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(10, 37, 53, 0.03);
        }
        [data-testid="stMetricLabel"] * { color: #4A8FA8 !important; font-weight: 600; }
        [data-testid="stMetricValue"] { color: #0A2535 !important; font-weight: 800; }

        /* --- BOUTONS DE LA SIDEBAR (export PDF, actions) --- */
        [data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            border: 1px solid #00C8E6 !important;
            background-color: rgba(0, 200, 230, 0.1) !important;
            border-radius: 6px;
        }
        [data-testid="stSidebar"] [data-testid="stDownloadButton"] button p,
        [data-testid="stSidebar"] [data-testid="stButton"] button p {
            color: #00C8E6 !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover,
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
            background-color: rgba(0, 200, 230, 0.25) !important;
        }

        /* --- BANDEAUX D'ALERTE VISUELLE --- */
        .alerte-danger  { border-left: 4px solid #C8294E; padding: 12px; background: rgba(255,255,255,0.8); margin-bottom: 10px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .alerte-warning { border-left: 4px solid #B07010; padding: 12px; background: rgba(255,255,255,0.8); margin-bottom: 10px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .alerte-success { border-left: 4px solid #007858; padding: 12px; background: rgba(255,255,255,0.8); margin-bottom: 10px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }

        .topbar-badge {
            background-color: #007858; color: white; padding: 4px 10px; border-radius: 12px;
            font-size: 11px; font-weight: bold; vertical-align: middle; margin-left: 10px;
        }

        /* --- ICONES PROFESSIONNELLES UTILISEES DANS LES POPUPS FOLIUM --- */
        .icon-alerte  { color: #C8294E; margin-right: 6px; }   /* Icone rouge : fuite */
        .icon-stable  { color: #007858; margin-right: 6px; }   /* Icone verte : etat stable */
        .icon-lien    { margin-right: 6px; }                   /* Icone du bouton Google Maps */

        /* --- PAGE DE CONNEXION --- */
        .login-titre { text-align:center; margin-top: 40px; }
        .login-souscritre { text-align:center; color:#4A8FA8; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# AUTHENTIFICATION (ACCES RESERVE AUX ADMINISTRATEURS)
# ==============================================================================
# L'application etant destinee a etre hebergee publiquement (Streamlit Community
# Cloud), elle ne doit PAS etre consultable par n'importe qui. Il n'y a AUCUN
# formulaire d'inscription : seuls des comptes admin crees a l'avance (par toi,
# directement dans les secrets de l'application) peuvent se connecter.
#
# Les identifiants sont stockes dans st.secrets, jamais dans le code source.
# Cree un fichier .streamlit/secrets.toml (en local, NE JAMAIS le versionner
# dans Git) avec la structure suivante :
#
#   [admins]
#   jonathan = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d"
#
# La valeur n'est PAS le mot de passe en clair, mais son empreinte SHA-256.
# Pour generer cette empreinte a partir d'un mot de passe choisi, execute :
#   python3 -c "import hashlib; print(hashlib.sha256('MonMotDePasse'.encode()).hexdigest())"
#
# Une fois deploye sur Streamlit Community Cloud, la meme structure se
# configure depuis l'interface "Secrets" du tableau de bord de l'application
# (Settings > Secrets), sans jamais toucher au code.
#
# NOTE D'EVOLUTION : cette fonction de verification est volontairement isolee.
# Le jour ou la base de donnees migrera vers Supabase, il suffira de remplacer
# son contenu par un appel a Supabase Auth (supabase.auth.sign_in_with_password)
# sans modifier le reste de l'application.
# ==============================================================================

def verifier_identifiants(nom_utilisateur: str, mot_de_passe: str) -> bool:
    """
    Verifie un couple nom d'utilisateur / mot de passe par rapport aux
    comptes admin declares dans st.secrets. Retourne True si l'authentification
    reussit, False sinon. Le mot de passe n'est jamais compare en clair : on
    compare son empreinte SHA-256 a celle enregistree dans les secrets.
    """
    admins = st.secrets.get("admins", {})
    if nom_utilisateur not in admins:
        return False
    empreinte_saisie = hashlib.sha256(mot_de_passe.encode()).hexdigest()
    empreinte_attendue = admins[nom_utilisateur]
    return empreinte_saisie == empreinte_attendue


def afficher_page_connexion():
    """
    Affiche un formulaire de connexion centre, dans le meme theme visuel
    (navy/aqua) que le reste de l'application. Tant que l'authentification
    n'a pas reussi, aucune autre partie de l'application (sidebar, donnees,
    cartographie...) n'est chargee ni affichee.
    """
    col_gauche, col_centre, col_droite = st.columns([1, 1.2, 1])
    with col_centre:
        st.markdown(
            "<h2 class='login-titre'> REGIDESO S.A.</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p class='login-souscritre'>Acces reserve aux administrateurs du systeme</p>",
            unsafe_allow_html=True
        )

        with st.form("formulaire_connexion", clear_on_submit=False):
            nom_utilisateur = st.text_input("Identifiant")
            mot_de_passe = st.text_input("Mot de passe", type="password")
            connexion_soumise = st.form_submit_button(
                ":material/login: Se connecter", use_container_width=True
            )

        if connexion_soumise:
            if verifier_identifiants(nom_utilisateur, mot_de_passe):
                st.session_state["authentifie"] = True
                st.session_state["utilisateur_connecte"] = nom_utilisateur
                st.rerun()
            else:
                st.error(":material/error: Identifiant ou mot de passe incorrect.")


# Initialisation de l'etat de connexion (une seule fois par session utilisateur)
if "authentifie" not in st.session_state:
    st.session_state["authentifie"] = False

# GARDE D'ACCES : tant que l'utilisateur n'est pas authentifie, on affiche
# uniquement la page de connexion et on arrete completement l'execution du
# script ici. Aucune donnee, aucune sidebar, aucun tableau de bord n'est
# charge ou expose a un visiteur non authentifie.
if not st.session_state["authentifie"]:
    afficher_page_connexion()
    st.stop()

# ==============================================================================
# COORDONNEES GEOGRAPHIQUES DE REFERENCE PAR COMMUNE (CENTRE APPROXIMATIF)
# ==============================================================================
COORDS_COMMUNES = {
    "Bandalungwa": {"lat": -4.336, "lon": 15.286},
    "Barumbu":     {"lat": -4.317, "lon": 15.322},
    "Gombe":       {"lat": -4.303, "lon": 15.309},
    "Kalamu":      {"lat": -4.348, "lon": 15.313},
    "Limete":      {"lat": -4.348, "lon": 15.334},
    "Lemba":       {"lat": -4.373, "lon": 15.328},
    "Ngaliema":    {"lat": -4.351, "lon": 15.251},
    "Kinshasa":    {"lat": -4.314, "lon": 15.311},
}


# ==============================================================================
# CHARGEMENT DES DONNEES DEPUIS LA BASE SQLITE
# ==============================================================================
DB_NAME = "regideso_comparatif.db"   # Nom du fichier de base de donnees local

def charger_donnees():
    """
    Se connecte a la base SQLite et recupere les deux tables necessaires :
    - mesures_comparatives : les mesures de debit/pression/alertes par commune
    - topologie_secteurs   : les informations structurelles du reseau
    Retourne deux DataFrames vides en cas d'echec (base absente, table manquante, etc.).
    """
    try:
        conn = sqlite3.connect(DB_NAME)                                      # Ouverture de la connexion
        df_sql = pd.read_sql_query("SELECT * FROM mesures_comparatives", conn)   # Lecture des mesures
        df_topo = pd.read_sql_query("SELECT * FROM topologie_secteurs", conn)    # Lecture de la topologie
        conn.close()                                                         # Fermeture propre de la connexion
        df_sql['timestamp'] = pd.to_datetime(df_sql['timestamp'])            # Conversion en type date/heure
        df_topo['timestamp'] = pd.to_datetime(df_topo['timestamp'])          # Idem pour la topologie
        return df_sql, df_topo
    except Exception:
        # En cas d'erreur (base absente, table inexistante...) on retourne des DataFrames vides
        return pd.DataFrame(), pd.DataFrame()

# Chargement effectif des donnees au demarrage du script
df, df_topo = charger_donnees()

# Si aucune donnee n'est disponible, on affiche un message et on arrete l'execution
if df.empty or df_topo.empty:
    st.markdown(
        "<div class='alerte-warning'><strong>:material/hourglass_empty: En attente du flux...</strong> "
        "Lancez le générateur de données.</div>",
        unsafe_allow_html=True
    )
    st.stop()   # Interrompt le script ici : inutile de continuer sans donnees


# ==============================================================================
# GENERATION DU RAPPORT PDF
# ==============================================================================
def compiler_rapport_pdf(df_c, df_t, modele, col_alerte, commune):
    """
    Construit un rapport PDF de synthese pour une commune donnee :
    - en-tete officiel REGIDESO
    - parametres de surveillance (commune, algorithme IA actif)
    - synthese telemetrique (pression moyenne, debit moyen, nombre d'anomalies)
    - journal des 5 dernieres anomalies detectees
    """
    pdf = FPDF()
    pdf.add_page()

    # --- En-tete principal ---
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(11, 32, 48)                                  # Couleur Navy
    pdf.cell(0, 12, "REGIDESO S.A.", ln=1, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(74, 143, 168)
    pdf.cell(0, 8, "Systeme Intelligent de Detection de Fuites (Aide a la Decision)", ln=1, align="C")
    pdf.cell(0, 6, f"Rapport genere automatiquement le : {datetime.now().strftime('%d/%m/%Y a %H:%M')}", ln=1, align="C")
    pdf.ln(8)

    # --- Ligne de separation Aqua ---
    pdf.set_draw_color(0, 200, 230)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # --- Section 1 : parametres du secteur surveille ---
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(11, 32, 48)
    pdf.cell(0, 10, "1. Parametres du Secteur Surveille", ln=1)

    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Commune / District :", ln=0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, str(commune), ln=1)

    pdf.set_font("Arial", "", 11)
    pdf.cell(50, 7, "Algorithme IA Operationnel :", ln=0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, str(modele), ln=1)
    pdf.ln(4)

    # --- Section 2 : synthese telemetrique globale ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "2. Synthese Telemetrique Globale", ln=1)

    # Calcul des indicateurs cles (avec garde-fous si colonnes absentes)
    pression_moy = df_c['pression_bar'].mean() if 'pression_bar' in df_c.columns else 0
    debit_moy = df_c['consommation_m3'].mean() if 'consommation_m3' in df_c.columns else 0
    total_anomalies = df_c[col_alerte].sum() if col_alerte in df_c.columns else 0

    pdf.set_font("Arial", "", 11)
    pdf.cell(70, 7, f"- Pression moyenne enregistree : {pression_moy:.2f} Bar", ln=1)
    pdf.cell(70, 7, f"- Debit moyen du secteur : {debit_moy:.1f} m3/h", ln=1)
    pdf.cell(70, 7, f"- Nombre total de points d'anomalies : {int(total_anomalies)}", ln=1)
    pdf.ln(6)

    # --- Bandeau visuel de gravite (rouge si anomalies, vert sinon) ---
    if total_anomalies > 0:
        pdf.set_fill_color(200, 41, 78)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 10, f" CRITIQUE : {int(total_anomalies)} fuite(s) suspecte(s) identifiee(s) par l'IA !", ln=1, fill=True)
    else:
        pdf.set_fill_color(0, 120, 88)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 10, " NORMAL : Le comportement du reseau est stable sur cette zone.", ln=1, fill=True)

    pdf.set_text_color(11, 32, 48)
    pdf.ln(6)

    # --- Section 3 : journal des dernieres anomalies (uniquement si anomalies presentes) ---
    if total_anomalies > 0 and col_alerte in df_c.columns:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "3. Journal de ciblage des anomalies recents", ln=1)

        # En-tete du tableau
        pdf.set_font("Arial", "B", 10)
        pdf.set_fill_color(13, 42, 63)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(60, 8, "Date & Heure", 1, 0, "C", True)
        pdf.cell(65, 8, "Debit suspect (m3/h)", 1, 0, "C", True)
        pdf.cell(60, 8, "Pression mesuree (Bar)", 1, 1, "C", True)

        # Corps du tableau : les 5 dernieres anomalies
        pdf.set_text_color(11, 32, 48)
        pdf.set_font("Arial", "", 10)
        df_anom = df_c[df_c[col_alerte] == 1].tail(5)
        for _, row in df_anom.iterrows():
            ts_str = row['timestamp'].strftime('%d/%m %H:%M:%S') if isinstance(row['timestamp'], datetime) else str(row['timestamp'])
            pdf.cell(60, 8, ts_str, 1, 0, "C")
            pdf.cell(65, 8, f"{row['consommation_m3']:.1f}", 1, 0, "C")
            pdf.cell(60, 8, f"{row['pression_bar']:.2f}", 1, 1, "C")

    # --- Export final en bytes (compatibilite selon version de fpdf2) ---
    out_pdf = pdf.output()
    return out_pdf.encode('latin1') if isinstance(out_pdf, str) else bytes(out_pdf)


# ==============================================================================
# ENTRAINEMENT PROPHET OPTIMISE (MISE EN CACHE + MODELE ALLEGE)
# ==============================================================================

# Duree de vie du cache : Prophet ne sera reentraine qu'une fois toutes les X secondes,
# meme si la page se rafraichit automatiquement toutes les 4 secondes (streaming_actif).
PROPHET_CACHE_TTL_SECONDES = 30

# Nombre maximal de points historiques utilises pour l'entrainement. Sans cette limite,
# le temps d'entrainement augmenterait indefiniment avec l'accumulation du flux de donnees.
PROPHET_FENETRE_MAX_POINTS = 300


@st.cache_data(ttl=PROPHET_CACHE_TTL_SECONDES, show_spinner=False)
def entrainer_prophet_optimise(_df_p, commune, cache_bucket, nb_lignes):
    """
    Entraine un modele Prophet allege et met le resultat en cache.

    Parametres de cache :
    - _df_p        : DataFrame source (prefixe par '_' => non hache par Streamlit,
                      on controle nous-memes quand invalider via cache_bucket/nb_lignes)
    - commune      : cle de cache pour ne pas partager le resultat entre communes
    - cache_bucket : entier qui change toutes les PROPHET_CACHE_TTL_SECONDES secondes,
                      force un reentrainement periodique meme si le flux continue d'arriver
    - nb_lignes    : invalide aussi le cache si le volume de donnees a change significativement

    Optimisations appliquees au modele :
    - desactivation des saisonnalites annuelle/hebdomadaire/journaliere : inutiles ici car
      les mesures sont espacees de quelques secondes, pas de cycle jour/semaine/annee a capter
    - uncertainty_samples=0 : desactive l'echantillonnage des intervalles de confiance,
      qui est l'etape la plus couteuse en temps de calcul de Prophet
    - fenetre limitee aux derniers points (voir PROPHET_FENETRE_MAX_POINTS en amont)
    """
    # On ne garde que les points les plus recents pour garder un entrainement rapide
    df_recent = _df_p.tail(PROPHET_FENETRE_MAX_POINTS).drop_duplicates(subset="ds").sort_values("ds")

    modele = Prophet(
        yearly_seasonality=False,   # pas de cycle annuel pertinent sur des donnees a l'echelle de la seconde
        weekly_seasonality=False,   # idem pour le cycle hebdomadaire
        daily_seasonality=False,    # idem pour le cycle journalier
        uncertainty_samples=0,      # desactive le calcul des intervalles de confiance (gain de perf majeur)
    )
    modele.fit(df_recent)

    futur = modele.make_future_dataframe(periods=10, freq='4s')
    prevision = modele.predict(futur)
    return prevision


# ==============================================================================
# POSITIONNEMENT DES FUITES SUR LE VRAI RESEAU ROUTIER (AVENUES/RUES REELLES)
# ==============================================================================

# Rayon (en metres) autour du centre de chaque commune dans lequel on telecharge le
# reseau routier reel. Doit couvrir la zone de dispersion utilisee par coord_point().
RAYON_RESEAU_ROUTIER_METRES = 1500


@st.cache_resource(show_spinner=False)
def charger_reseau_routier_commune(commune, lat_centre, lon_centre, rayon_metres=RAYON_RESEAU_ROUTIER_METRES):
    """
    Telecharge le graphe du reseau routier reel (OpenStreetMap) autour du centre d'une
    commune, et le met en cache pour toute la duree de vie du serveur (le telechargement
    ne se produit qu'une seule fois par commune, pas a chaque rafraichissement de la page).
    """
    return ox.graph_from_point((lat_centre, lon_centre), dist=rayon_metres, network_type="drive")


def positionner_sur_avenues_reelles(df_sous_ensemble, graphe):
    """
    Projette chaque position approximative (lat_approx, lon_approx) sur le noeud du reseau
    routier reel le plus proche, et recupere le nom de l'avenue/rue associee lorsque cette
    information existe dans les donnees OpenStreetMap. Le point affiche sur la carte se
    retrouve ainsi exactement sur une vraie avenue au lieu d'un emplacement aleatoire.
    """
    # Recherche vectorisee (en un seul appel) du noeud le plus proche pour chaque ligne
    noeuds = ox.distance.nearest_nodes(
        graphe,
        X=df_sous_ensemble['lon_approx'].tolist(),
        Y=df_sous_ensemble['lat_approx'].tolist()
    )

    # Construction d'une correspondance noeud -> nom de rue a partir des segments du graphe
    nom_rue_par_noeud = {}
    for u, v, data in graphe.edges(data=True):
        nom = data.get('name', 'Avenue non identifiée')
        nom = nom[0] if isinstance(nom, list) else nom   # certaines rues ont plusieurs noms dans OSM
        nom_rue_par_noeud.setdefault(u, nom)
        nom_rue_par_noeud.setdefault(v, nom)

    # Extraction des coordonnees reelles et du nom d'avenue pour chaque noeud trouve
    df_resultat = df_sous_ensemble.copy()
    df_resultat['lat_point'] = [graphe.nodes[n]['y'] for n in noeuds]
    df_resultat['lon_point'] = [graphe.nodes[n]['x'] for n in noeuds]
    df_resultat['nom_avenue'] = [nom_rue_par_noeud.get(n, 'Avenue non identifiée') for n in noeuds]
    return df_resultat


# ==============================================================================
# CONSTRUCTION DE LA SIDEBAR (UNIQUE - SANS DOUBLON)
# ==============================================================================
with st.sidebar:

    # --- Bloc 1 : logo et identite de l'application ---
    col_l1, col_l2 = st.columns([1, 3], vertical_alignment="center")
    with col_l1:
        try:
            st.image(Image.open('assets/logo_regideso.png'), width=45)   # Logo officiel si present
        except Exception:
            st.markdown("<h2 style='color:#00C8E6; margin:0;'>:material/water_drop:</h2>", unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
            <div style='line-height:1.1;'>
                <strong style='color:#EDF5F9; font-size:15px; letter-spacing:0.5px;'>REGIDESO S.A.</strong><br>
                <span style='color:#4A8FA8; font-size:10px;'>Système IA · Kinshasa</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    st.text_input("Recherche", placeholder="Rechercher une zone...", label_visibility="collapsed")
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # --- Bloc 2 : menu de navigation principal ---
    options_nav = [
        ":material/dashboard: Vue d'ensemble",
        ":material/activity_zone: Surveillance Live",
        ":material/map: Cartographie SIG",
        ":material/compare_arrows: Duel d'Algorithmes",
        ":material/online_prediction: Maintenance Prédictive",
        ":material/rule: Scoring des Risques",
        ":material/account_balance: Impact Financier",
        ":material/notifications_active: Centre d'Alertes",
    ]
    navigation = st.radio("Navigation", options=options_nav, label_visibility="collapsed")

    st.markdown(
        "<div style='border-top:1px solid rgba(0,200,230,0.1); margin-top:15px; margin-bottom:15px;'></div>",
        unsafe_allow_html=True
    )

    # --- Bloc 3 : filtres de secteur et parametres globaux ---
    communes_dispo = sorted(df['commune'].unique())                    # Liste triee des communes disponibles
    commune_choisie = st.selectbox("Secteur ciblé", communes_dispo, key="sb_secteur")   # Commune active

    modele_actif = st.radio(
        "Moteur IA Actif", ["Isolation Forest", "One-Class SVM"],
        horizontal=True, key="rb_moteur"
    )
    # Selection de la colonne d'alerte correspondant au modele choisi
    col_alerte_modele = "alerte_iforest" if modele_actif == "Isolation Forest" else "alerte_ocsvm"

    streaming_actif = st.toggle("Flux Temps Réel", value=True, key="tg_streaming")   # Active/desactive l'auto-refresh

    # --- Bloc 4 : export du rapport PDF (une seule fois, plus de doublon) ---
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    try:
        df_commune = df[df['commune'] == commune_choisie]   # Filtrage des donnees sur la commune active
        pdf_bytes = compiler_rapport_pdf(df_commune, df_topo, modele_actif, col_alerte_modele, commune_choisie)
        st.download_button(
            label=":material/picture_as_pdf: Exporter le Rapport PDF",
            data=pdf_bytes,
            file_name=f"Rapport_REGIDESO_{commune_choisie}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Erreur PDF : {e}")

    # --- Bloc 5 : identite de l'admin connecte + deconnexion ---
    st.markdown(
        "<div style='border-top:1px solid rgba(0,200,230,0.1); margin-top:15px; margin-bottom:15px;'></div>",
        unsafe_allow_html=True
    )
    st.caption(f":material/person: Connecté en tant que **{st.session_state.get('utilisateur_connecte', '')}**")
    if st.button(":material/logout: Se déconnecter", use_container_width=True, key="btn_deconnexion"):
        st.session_state["authentifie"] = False
        st.session_state.pop("utilisateur_connecte", None)
        st.rerun()


# ==============================================================================
# ZONE PRINCIPALE - CONTENU CENTRAL DE L'APPLICATION
# ==============================================================================

# Filtrage global des donnees sur la commune actuellement selectionnee (reutilise dans plusieurs onglets)
df_commune = df[df['commune'] == commune_choisie]

# --- Bandeau superieur (topbar) ---
st.markdown("""
    <div style='background-color: rgba(255,255,255,0.80); padding: 10px 20px; border-radius: 8px;
                border: 1px solid rgba(0,160,190,0.14); margin-bottom: 20px;
                display: flex; justify-content: space-between; align-items: center;'>
        <span style='color: #4A8FA8; font-weight: 600; font-size: 13px;'>RÉSEAU PRIMAIRE KINSHASA</span>
        <div><span class='topbar-badge'>:material/check_circle: Système Actif</span></div>
    </div>
""", unsafe_allow_html=True)

# --- Titre principal ---
st.markdown(
    "<h1 style='text-align: center;'>Système Intelligent de "
    "<span style='color: #007A96;'>Détection de Fuites</span></h1>",
    unsafe_allow_html=True
)
st.write("---")


# ==============================================================================
# ROUTAGE DES PAGES SELON L'ONGLET DE NAVIGATION SELECTIONNE
# ==============================================================================

# Le contenu de chaque page est regroupe dans une fonction unique. Cela permet de
# l'envelopper avec st.fragment (voir plus bas) : seule cette fonction est reexecutee
# automatiquement toutes les 4 secondes en mode streaming, jamais la sidebar ni le reste
# de la page. Cela evite tout effet de clignotement ou de superposition visuelle entre
# onglets pendant le rafraichissement.
def afficher_contenu_principal():


    # ------------------------------------------------------------------------
    # PAGE : VUE D'ENSEMBLE
    # ------------------------------------------------------------------------
    if "Vue d'ensemble" in navigation:

        # --- Fenetre glissante utilisee pour rendre le tableau de bord vraiment "temps reel" ---
        # Sans cette fenetre, les moyennes/compteurs sont calcules sur TOUT l'historique de la
        # commune : ajouter un point toutes les 4 secondes a une moyenne de plusieurs milliers de
        # lignes ne bouge quasiment rien a l'affichage, d'ou l'impression que rien ne se passe.
        # On limite donc l'analyse aux N dernieres mesures reellement recentes de la commune.
        FENETRE_VUE_ENSEMBLE = 40   # ~ 2 a 3 minutes d'historique recent a raison d'un point / 4s

        # Tri chronologique puis decoupage en deux fenetres consecutives :
        # - fenetre_actuelle  : les N dernieres mesures (etat "maintenant")
        # - fenetre_precedente: les N mesures juste avant (sert de reference pour calculer un delta)
        df_commune_triee = df_commune.sort_values('timestamp')
        fenetre_actuelle = df_commune_triee.tail(FENETRE_VUE_ENSEMBLE)
        fenetre_precedente = df_commune_triee.iloc[-(2 * FENETRE_VUE_ENSEMBLE):-FENETRE_VUE_ENSEMBLE]

        # --- Bloc de metriques cles pour la commune selectionnee (recalcule sur la fenetre recente) ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Communes Supervisées", len(communes_dispo))

        # Pression moyenne recente + comparaison avec la fenetre precedente (delta visible a chaque refresh)
        pression_actuelle = fenetre_actuelle['pression_bar'].mean() if not fenetre_actuelle.empty else 0
        pression_precedente = fenetre_precedente['pression_bar'].mean() if not fenetre_precedente.empty else pression_actuelle
        ecart_pression = pression_actuelle - pression_precedente
        c2.metric(
            f"Pression Moy. Récente ({commune_choisie})",
            f"{pression_actuelle:.2f} Bar",
            delta=f"{ecart_pression:+.2f} Bar vs période précédente"
        )

        # Anomalies sur la fenetre recente (indicateur reactif) + rappel du cumul historique en petit texte
        anomalies_recentes = fenetre_actuelle[col_alerte_modele].sum() if not fenetre_actuelle.empty else 0
        anomalies_precedentes = fenetre_precedente[col_alerte_modele].sum() if not fenetre_precedente.empty else 0
        ecart_anomalies = int(anomalies_recentes - anomalies_precedentes)
        c3.metric(
            label="Anomalies (fenêtre récente)",
            value=int(anomalies_recentes),
            delta=f"{ecart_anomalies:+d} vs période précédente" if ecart_anomalies != 0 else "Stable sur la période",
            delta_color="inverse" if ecart_anomalies > 0 else "normal"
        )
        total_anomalies_historique = df_commune[col_alerte_modele].sum() if not df_commune.empty else 0
        st.caption(f"Cumul historique pour {commune_choisie} : {int(total_anomalies_historique)} anomalie(s) détectée(s) au total.")

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        # --- Courbe temps reel de la pression sur la fenetre recente : montre visuellement
        #     l'arrivee de nouveaux points a chaque rafraichissement, propre a chaque commune ---
        fig_tendance = px.line(
            fenetre_actuelle,
            x="timestamp",
            y="pression_bar",
            title=f"Tendance de Pression en Temps Réel — Secteur {commune_choisie}",
            color_discrete_sequence=["#00C8E6"]
        )
        points_alerte = fenetre_actuelle[fenetre_actuelle[col_alerte_modele] == 1]
        fig_tendance.add_scatter(
            x=points_alerte['timestamp'], y=points_alerte['pression_bar'],
            mode='markers', name='Anomalie', marker=dict(color='#C8294E', size=9, symbol='x')
        )
        fig_tendance.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', template="plotly_dark")
        st.plotly_chart(fig_tendance, use_container_width=True)

        # --- Histogramme de la distribution des pressions, calcule uniquement sur la fenetre
        #     recente : la forme change reellement d'un rafraichissement a l'autre et differe
        #     naturellement d'une commune a l'autre puisque chacune a son propre sous-ensemble ---
        fig = px.histogram(
            fenetre_actuelle,
            x="pression_bar",
            color=col_alerte_modele,
            title=f"Distribution des Pressions (fenêtre récente) — Secteur {commune_choisie}",
            color_discrete_sequence=["#0088AA", "#C8294E"]
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------------------------------------
    # PAGE : SURVEILLANCE LIVE
    # ------------------------------------------------------------------------
    elif "Surveillance Live" in navigation:
        st.subheader(f":material/activity_zone: Métriques en direct — {commune_choisie}")

        # --- Courbe de debit avec superposition des points d'anomalies ---
        fig = px.line(
            df_commune, x='timestamp', y='consommation_m3',
            title="Variations de Débit (m³/h)", color_discrete_sequence=["#0088AA"]
        )
        anomalies = df_commune[df_commune[col_alerte_modele] == 1]
        fig.add_scatter(
            x=anomalies['timestamp'], y=anomalies['consommation_m3'],
            mode='markers', name='Fuite Détectée',
            marker=dict(color='#C8294E', size=10, symbol='x')
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # --- Indicateurs instantanes (derniere mesure recue) ---
        if not df_commune.empty:
            k1, k2 = st.columns(2)
            k1.metric("Débit Instantané", f"{df_commune['consommation_m3'].iloc[-1]:.1f} m³/h")
            k2.metric("Pression Aval", f"{df_commune['pression_bar'].iloc[-1]:.2f} Bar")
        else:
            st.warning("En attente de réception des premières données télémétriques...")


    # ------------------------------------------------------------------------
    # PAGE : CARTOGRAPHIE SIG (VUE GLOBALE DE TOUTE LA CAPITALE)
    # ------------------------------------------------------------------------
    elif "Cartographie SIG" in navigation:
        st.subheader(":material/map: Cartographie des Fuites — Ville de Kinshasa")
        st.caption("Vue globale de toutes les communes supervisées. Cliquez sur un point pour voir les caractéristiques de la fuite.")

        # --- IMPORTANT : on utilise TOUTE la base (df), pas seulement la commune selectionnee,
        #     afin d'afficher l'etat complet de la capitale en un seul coup d'oeil. ---
        df_carte = df.copy()

        # On ne garde que les 30 dernieres mesures par commune pour garder une carte lisible
        df_carte = df_carte.sort_values('timestamp').groupby('commune').tail(30)

        def jitter_seed(commune, idx):
            """Genere un seed deterministe (stable) a partir de la commune et de l'index de la ligne."""
            key = f"{commune}_{idx}".encode()
            return int(hashlib.md5(key).hexdigest(), 16) % (2**32 - 1)

        def coord_point_approximative(row):
            """
            Calcule une position APPROXIMATIVE (pas encore alignee sur une vraie rue) dispersee
            autour du centre de la commune. Cette position sert uniquement de point de depart
            pour la recherche de l'avenue reelle la plus proche (voir positionner_sur_avenues_reelles).
            """
            base = COORDS_COMMUNES.get(row['commune'], {"lat": -4.325, "lon": 15.322})
            rng = np.random.RandomState(jitter_seed(row['commune'], row.name))
            lat = base['lat'] + rng.uniform(-0.010, 0.010)
            lon = base['lon'] + rng.uniform(-0.010, 0.010)
            return pd.Series({'lat_approx': lat, 'lon_approx': lon})

        # Calcul de la position approximative de depart pour chaque ligne
        df_carte[['lat_approx', 'lon_approx']] = df_carte.apply(coord_point_approximative, axis=1)

        # Position par defaut = position approximative, tant qu'aucun aimantage n'a reussi
        df_carte['lat_point'] = df_carte['lat_approx']
        df_carte['lon_point'] = df_carte['lon_approx']
        df_carte['nom_avenue'] = "Position approximative (avenue non confirmée)"

        # --- Aimantage sur le vrai reseau routier OpenStreetMap, commune par commune ---
        # On traite chaque commune separement : le graphe routier est plus petit (donc rapide a
        # charger) et mis en cache individuellement, si bien qu'il n'est telecharge qu'une fois.
        au_moins_une_commune_en_mode_approximatif = not OSMNX_DISPONIBLE

        if OSMNX_DISPONIBLE:
            for commune, groupe in df_carte.groupby('commune'):
                centre = COORDS_COMMUNES.get(commune, {"lat": -4.325, "lon": 15.322})
                try:
                    # Chargement (ou reutilisation depuis le cache) du reseau routier reel de la commune
                    graphe_commune = charger_reseau_routier_commune(commune, centre['lat'], centre['lon'])
                    # Projection des points approximatifs sur les vraies avenues du reseau
                    groupe_aimante = positionner_sur_avenues_reelles(groupe, graphe_commune)
                    df_carte.loc[groupe_aimante.index, ['lat_point', 'lon_point', 'nom_avenue']] = \
                        groupe_aimante[['lat_point', 'lon_point', 'nom_avenue']]
                except Exception:
                    # Reseau OSM injoignable ou zone non couverte : on garde la position approximative
                    # deja en place pour cette commune, sans faire echouer le reste de la carte.
                    au_moins_une_commune_en_mode_approximatif = True

        if au_moins_une_commune_en_mode_approximatif:
            st.info(
                ":material/info: Positionnement exact sur les avenues indisponible pour au moins une zone "
                "(réseau routier OpenStreetMap non joignable ou librairie osmnx absente) — "
                "affichage en position approximative pour ces points."
            )

        # --- Creation de la carte Folium centree sur l'ensemble de Kinshasa ---
        carte = folium.Map(location=[-4.325, 15.322], zoom_start=11, tiles="CartoDB positron")

        # --- Ajout d'un marqueur individuel par mesure/fuite ---
        for _, row in df_carte.iterrows():
            est_fuite = row[col_alerte_modele] == 1
            couleur = "#C8294E" if est_fuite else "#007858"
            lien_gmaps = f"https://www.google.com/maps?q={row['lat_point']},{row['lon_point']}"

            # Icone professionnelle (Font Awesome, deja embarque par Folium) au lieu d'un emoji
            icone_statut = (
                '<i class="fa fa-exclamation-triangle icon-alerte"></i> Fuite suspectée'
                if est_fuite else
                '<i class="fa fa-check-circle icon-stable"></i> Point stable'
            )

            # Contenu HTML du popup affiche au clic sur le point (avec le nom de l'avenue reelle)
            popup_html = f"""
            <div style='font-family: Arial; font-size:13px; width:240px;'>
                <b style='color:#0A2535;'>{icone_statut}</b><br>
                <b>Commune :</b> {row['commune']}<br>
                <b>Avenue / Rue :</b> {row['nom_avenue']}<br>
                <b>Date :</b> {row['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}<br>
                <b>Débit :</b> {row['consommation_m3']:.1f} m³/h<br>
                <b>Pression :</b> {row['pression_bar']:.2f} Bar<br>
                <a href="{lien_gmaps}" target="_blank"
                   style='display:inline-block; margin-top:6px; padding:5px 10px; background:#00C8E6;
                          color:white; border-radius:4px; text-decoration:none; font-weight:bold;'>
                    <i class="fa fa-map-marker-alt icon-lien"></i>Voir sur Google Maps
                </a>
            </div>
            """

            # Marqueur circulaire colore selon le statut (rouge = fuite, vert = stable)
            folium.CircleMarker(
                location=[row['lat_point'], row['lon_point']],
                radius=8 if est_fuite else 5,
                color=couleur,
                fill=True,
                fill_color=couleur,
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=f"{row['commune']} — {row['nom_avenue']}"
            ).add_to(carte)

        # Affichage de la carte dans Streamlit (returned_objects=[] evite de renvoyer trop de donnees a chaque interaction)
        st_folium(carte, width=None, height=520, returned_objects=[], key="carte_kinshasa")


    # ------------------------------------------------------------------------
    # PAGE : DUEL D'ALGORITHMES (BENCHMARK ML)
    # ------------------------------------------------------------------------
    elif "Duel d'Algorithmes" in navigation:
        st.subheader(":material/compare_arrows: Benchmark ML")

        # --- Tableau comparatif des performances (valeurs de reference / demonstration) ---
        df_metrics = pd.DataFrame({
            "Modèle": ["Isolation Forest", "One-Class SVM"],
            "Précision": [0.92, 0.81],
            "Rappel": [0.89, 0.76],
            "F1-Score": [0.90, 0.78]
        })
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # --- Courbes ROC comparatives des deux modeles ---
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 0.05, 1], y=[0, 0.94, 1], mode='lines', line=dict(color='#00C8E6', width=3), name='Isolation Forest'))
        fig_roc.add_trace(go.Scatter(x=[0, 0.15, 1], y=[0, 0.82, 1], mode='lines', line=dict(color='#4A8FA8', width=3), name='One-Class SVM'))
        fig_roc.update_layout(plot_bgcolor='rgba(255,255,255,0.55)', paper_bgcolor='rgba(0,0,0,0)', title="Courbe ROC Comparative")
        st.plotly_chart(fig_roc, use_container_width=True)


    # ------------------------------------------------------------------------
    # PAGE : MAINTENANCE PREDICTIVE (PROPHET)
    # ------------------------------------------------------------------------
    elif "Maintenance Prédictive" in navigation:
        st.subheader(":material/online_prediction: Prévision Prophet")

        # Il faut un minimum de points pour qu'un modele de serie temporelle soit pertinent
        if len(df_commune) > 5:
            try:
                # Prophet exige des colonnes nommees 'ds' (date) et 'y' (valeur)
                df_p = df_commune[['timestamp', 'consommation_m3']].rename(columns={'timestamp': 'ds', 'consommation_m3': 'y'})

                # Cle de cache temporelle : change toutes les PROPHET_CACHE_TTL_SECONDES secondes.
                # Tant que ce nombre ne change pas, l'appel suivant renvoie le resultat mis en cache
                # au lieu de relancer un entrainement complet a chaque rafraichissement de 4 secondes.
                cache_bucket = int(time.time() // PROPHET_CACHE_TTL_SECONDES)

                with st.spinner("Génération des prévisions..."):
                    prev = entrainer_prophet_optimise(df_p, commune_choisie, cache_bucket, len(df_p))

                # Petit indicateur discret pour informer sur la cadence reelle de reentrainement
                st.caption(f"Modèle réentraîné au plus toutes les {PROPHET_CACHE_TTL_SECONDES} secondes "
                           f"(fenêtre glissante des {PROPHET_FENETRE_MAX_POINTS} dernières mesures).")

                fig_prev = go.Figure()
                fig_prev.add_trace(go.Scatter(
                    x=prev['ds'].iloc[:-10], y=prev['yhat'].iloc[:-10],
                    mode='lines', line=dict(color='#4A8FA8'), name="Historique"
                ))
                fig_prev.add_trace(go.Scatter(
                    x=prev['ds'].iloc[-10:], y=prev['yhat'].iloc[-10:],
                    mode='lines', line=dict(color='#0088AA', width=3, dash='dash'), name="Projection Future"
                ))
                fig_prev.update_layout(plot_bgcolor='rgba(255,255,255,0.55)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_prev, use_container_width=True)

            except Exception as e:
                # Protege l'app contre un echec ponctuel de Prophet (donnees insuffisantes,
                # valeurs constantes, timestamps dupliques, etc.) sans faire planter le dashboard
                st.warning(f"Prévision momentanément indisponible : {e}")
        else:
            st.info("Attente de données suffisantes pour Prophet.")


    # ------------------------------------------------------------------------
    # PAGE : SCORING DES RISQUES
    # ------------------------------------------------------------------------
    elif "Scoring des Risques" in navigation:
        st.subheader(":material/rule: Priorisation des interventions")
        st.markdown(
            "<div class='alerte-warning'>Identification des secteurs critiques basés sur la vétusté et les anomalies.</div>",
            unsafe_allow_html=True
        )

        # Scores de risque simules (a remplacer par un vrai calcul base sur l'age du reseau + frequence d'anomalies)
        df_priorite = pd.DataFrame({'Zone': communes_dispo[:5], 'Score_Risque': [8.5, 7.2, 5.1, 4.0, 2.5]})

        fig_bar = px.bar(
            df_priorite, x='Score_Risque', y='Zone', orientation='h',
            color='Score_Risque', color_continuous_scale=["#0088AA", "#C8294E"]
        )
        fig_bar.update_layout(plot_bgcolor='rgba(255,255,255,0.55)', paper_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)


    # ------------------------------------------------------------------------
    # PAGE : IMPACT FINANCIER (EAU NON FACTUREE)
    # ------------------------------------------------------------------------
    elif "Impact Financier" in navigation:
        st.subheader(":material/account_balance: Pertes par Eau Non Facturée (ENF)")

        # Hypotheses de calcul (a remplacer par les vraies donnees financieres de la REGIDESO)
        perte_m3h = 14.5
        cout_usd = perte_m3h * 0.85 * 24

        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Manque à gagner / 24h", f"{cout_usd:,.2f} USD", delta="- Urgence financière", delta_color="inverse")
        col_f2.metric("Volume Perdu", f"{perte_m3h * 24:,.1f} m³")

        st.markdown(
            "<div class='alerte-danger'><strong>:material/info: Projection :</strong> "
            f"L'absence d'intervention coûtera {cout_usd * 7:,.2f} USD cette semaine.</div>",
            unsafe_allow_html=True
        )


    # ------------------------------------------------------------------------
    # PAGE : CENTRE D'ALERTES (DISPATCH DES INCIDENTS)
    # ------------------------------------------------------------------------
    elif "Centre d'Alertes" in navigation:
        st.subheader(":material/notifications_active: Dispatching des Incidents")

        mail_dest = f"resp.{commune_choisie.lower()}@regideso.cd"
        st.text_input("Ingénieur Responsable", value=mail_dest, disabled=True)

        if st.button(":material/send: Déclencher le protocole d'urgence"):
            with st.status("Transmission en cours..."):
                time.sleep(0.5)
                st.write(f"Ordre de mission envoyé à `{mail_dest}`")
            st.markdown(
                "<div class='alerte-success'><strong>:material/check_circle: Validé</strong> - "
                "L'équipe de terrain est notifiée.</div>",
                unsafe_allow_html=True
            )


# ==============================================================================
# EXECUTION DU CONTENU + RAFRAICHISSEMENT AUTOMATIQUE (MODE FLUX TEMPS REEL)
# ==============================================================================
# st.fragment isole le rafraichissement automatique : seule la fonction
# afficher_contenu_principal() est reexecutee toutes les 4 secondes, jamais la
# sidebar, le bandeau superieur ou le titre. Fini le clignotement de toute la page.
if hasattr(st, "fragment"):
    # Cle "run_every" : delai en secondes entre deux reexecutions automatiques du
    # fragment. On la desactive (None) si le streaming est coupe par l'utilisateur.
    afficher_contenu_principal = st.fragment(
        run_every=4 if streaming_actif else None
    )(afficher_contenu_principal)
    afficher_contenu_principal()
else:
    # Repli pour les versions anciennes de Streamlit qui n'ont pas encore st.fragment :
    # on reproduit l'ancien comportement (rafraichissement de toute la page).
    afficher_contenu_principal()
    if streaming_actif:
        time.sleep(4)
        st.rerun()