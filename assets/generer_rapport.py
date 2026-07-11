import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from weasyprint import HTML

def generer_rapport_pdf(date_cible="2026-06-12 08:00:00"):
    # 1. Connexion à la base de données et extraction
    conn = sqlite3.connect("regideso_comparatif.db")
    df_mesures = pd.read_sql_query("SELECT * FROM mesures_comparatives WHERE timestamp = ?", conn, params=(date_cible,))
    df_topo = pd.read_sql_query("SELECT * FROM topologie_secteurs WHERE timestamp = ?", conn, params=(date_cible,))
    conn.close()
    
    if df_mesures.empty or df_topo.empty:
        print("❌ Aucune donnée trouvée pour ce timestamp.")
        return

    # 2. Calcul des indicateurs clés (KPI)
    total_communes = len(df_mesures)
    total_anomalies = int(df_mesures['alerte_iforest'].sum())
    perte_totale_m3 = float(df_topo['perte_calculee'].sum())
    pression_moyenne = float(df_mesures['pression_bar'].mean()) # <- Définie ici
    
    df_critique = pd.merge(df_mesures, df_topo, on=['timestamp', 'commune'])
    df_critique = df_critique[df_critique['alerte_iforest'] == 1].sort_values(by='perte_calculee', ascending=False)

    # 3. Création du Graphique 1 : Diagramme circulaire de l'état du réseau
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    labels = ['Sain (Normal)', 'Anomalie (Fuite)']
    sizes = [total_communes - total_anomalies, total_anomalies]
    ax.pie(sizes, explode=(0, 0.08), labels=labels, colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=140)
    fig.gca().add_artist(plt.Circle((0,0), 0.55, fc='white')) # Effet Donut
    ax.set_title("État Global des 24 Communes", fontsize=11, fontweight='bold', color='#1A365D', pad=15)
    plt.tight_layout()
    
    buf1 = BytesIO()
    plt.savefig(buf1, format='png', dpi=200)
    buf1.seek(0)
    img_pie = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close()

    # 4. Création du Graphique 2 : Classement des pertes par commune
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(df_critique['commune'], df_critique['perte_calculee'], color='#E74C3C', height=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Volume perdu (m³/h)", fontsize=9, fontweight='bold')
    ax.set_title("Pertes par Commune Critique", fontsize=11, fontweight='bold', color='#1A365D', pad=15)
    plt.tight_layout()
    
    buf2 = BytesIO()
    plt.savefig(buf2, format='png', dpi=200)
    buf2.seek(0)
    img_bars = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close()

    # 5. Formatage de la date pour l'en-tête
    date_formatee = datetime.strptime(date_cible, "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y à %H:%M")

    # 6. Template HTML structurel (Bannissement de Flexbox/Grid pour WeasyPrint)
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 20mm 15mm;
                @bottom-right {{ 
                    content: "Page " counter(page) " sur " counter(pages); 
                    font-family: Arial; 
                    font-size: 8pt; 
                    color: #718096; 
                }}
            }}
            body {{ font-family: Arial, sans-serif; color: #2D3748; line-height: 1.4; }}
            .header-table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; border-bottom: 3px solid #1A365D; }}
            .logo-text {{ font-size: 20pt; font-weight: bold; color: #1A365D; }}
            .doc-title {{ font-size: 13pt; font-weight: bold; color: #2B6CB0; text-align: right; text-transform: uppercase; }}
            h2 {{ font-size: 12pt; color: #1A365D; border-left: 4px solid #2B6CB0; padding-left: 8px; margin-top: 25px; page-break-after: avoid; }}
            .kpi-table {{ width: 100%; border-collapse: separate; border-spacing: 10px 0; }}
            .kpi-card {{ display: table-cell; width: 25%; background-color: #F7FAFC; border: 1px solid #E2E8F0; border-radius: 5px; padding: 12px; text-align: center; }}
            .kpi-val {{ font-size: 16pt; font-weight: bold; color: #1A365D; }}
            .kpi-label {{ font-size: 8pt; color: #4A5568; font-weight: bold; text-transform: uppercase; margin-top: 4px; }}
            .charts-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .chart-cell {{ width: 50%; text-align: center; padding: 5px; }}
            .chart-img {{ max-width: 100%; height: auto; border: 1px solid #EDF2F7; border-radius: 4px; }}
            .data-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .data-table th {{ background-color: #1A365D; color: white; padding: 8px; font-size: 9pt; text-align: left; }}
            .data-table td {{ padding: 8px; border-bottom: 1px solid #E2E8F0; font-size: 9pt; }}
            .data-table tr:nth-child(even) {{ background-color: #F8FAFC; }}
            .badge {{ padding: 2px 6px; font-size: 7.5pt; font-weight: bold; border-radius: 3px; color: #9B2C2C; background-color: #FED7D7; }}
            .signature-table {{ width: 100%; margin-top: 40px; border-collapse: collapse; page-break-inside: avoid; }}
        </style>
    </head>
    <body>
        <table class="header-table">
            <tr>
                <td><div class="logo-text">💧 REGIDESO S.A.</div></td>
                <td><div class="doc-title">Rapport de Diagnostic Hydraulique</div><div style="text-align:right; font-size:8pt; color:#718096;">Date : {date_formatee}</div></td>
            </tr>
        </table>

        <h2>1. Indicateurs Synthétiques</h2>
        <table class="kpi-table">
            <tr>
                <td class="kpi-card"><div class="kpi-val">{total_communes}</div><div class="kpi-label">Secteurs</div></td>
                <td class="kpi-card" style="border-top: 4px solid #E53E3E;"><div class="kpi-val" style="color:#C53030;">{total_anomalies}</div><div class="kpi-label">Alertes Fuite</div></td>
                <td class="kpi-card" style="border-top: 4px solid #E53E3E;"><div class="kpi-val">{perte_totale_m3:.1f}</div><div class="kpi-label">Pertes Vol. (m³/h)</div></td>
                <td class="kpi-card"><div class="kpi-val">{pression_moyenne:.2f}</div><div class="kpi-label">Pression Moy. (Bar)</div></td>
            </tr>
        </table>

        <h2>2. Analyse Visuelle Réseau</h2>
        <table class="charts-table">
            <tr>
                <td class="chart-cell"><img class="chart-img" src="data:image/png;base64,{img_pie}"></td>
                <td class="chart-cell"><img class="chart-img" src="data:image/png;base64,{img_bars}"></td>
            </tr>
        </table>

        <h2>3. Localisation Opérationnelle des Fuites</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Commune</th>
                    <th>Statut Réseau</th>
                    <th>Débit Entrée</th>
                    <th>Pertes Estimées</th>
                    <th>Pression Constatée</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df_critique.iterrows():
        html_template += f"""
                <tr>
                    <td><strong>{row['commune']}</strong></td>
                    <td><span class="badge">Fuite Confirmée</span></td>
                    <td>{row['debit_source']:.1f} m³/h</td>
                    <td style="color:#C53030; font-weight:bold;">{row['perte_calculee']:.1f} m³/h</td>
                    <td>{row['pression_bar']:.2f} bar</td>
                </tr>
        """

    html_template += f"""
            </tbody>
        </table>

        <table class="signature-table">
            <tr>
                <td>
                    <p style="font-weight:bold; color:#1A365D;">L'analyste Projet IA,</p>
                    <br><br>
                    <p><strong>MULEBO WA MULEBO Jonathan</strong><br><span style="font-size:8pt; color:#4A5568;">Stagiaire Direction Digitalisation (UPC)</span></p>
                </td>
                <td style="text-align:right; vertical-align:top;">
                    <p style="font-weight:bold; color:#1A365D;">Validation Direction,</p>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    # Compilation du PDF
    HTML(string=html_template).write_pdf("Rapport_Journalier_REGIDESO.pdf")
    print("✨ Rapport PDF généré avec succès.")

# Test de la fonction
if __name__ == "__main__":
    generer_rapport_pdf()