import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import requests
import zipfile
import json
import io
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration du style (optionnel, laissé car bibliothèque graphique demandée)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==============================================================================
# --- CONFIGURATION ET CONSTANTES ---
# ==============================================================================

START_YEAR = 2002
END_YEAR = datetime.now().year
NVD_FEED_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{}.json.zip"
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.csv"

# ==============================================================================
# --- FONCTIONS D'ACQUISITION DE DONNÉES (ETL) ---
# ==============================================================================

def process_year_data(year):
    """
    Télécharge et extrait les données de vulnérabilités NVD pour une année spécifique.
    Retourne une liste de dictionnaires.
    """
    url = NVD_FEED_URL.format(year)
    year_cves = []
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            return []
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            json_filename = z.namelist()[0]
            with z.open(json_filename) as f:
                data = json.load(f)
                
        for cve_item in data.get('vulnerabilities', []):
            cve = cve_item.get('cve')
            cve_id = cve.get('id')
            
            # Récupération de la description anglaise
            description = ""
            for desc_item in cve.get('descriptions', []):
                if desc_item.get('lang') == 'en':
                    description = desc_item.get('value')
                    break
            
            published_date = cve.get('published')
            
            # Extraction du score CVSS (priorité v3.1 > v3.0)
            cvss_score = None
            metrics = cve.get('metrics', {})
            if 'cvssMetricV31' in metrics and metrics['cvssMetricV31']:
                cvss_score = metrics['cvssMetricV31'][0]['cvssData']['baseScore']
            elif 'cvssMetricV30' in metrics and metrics['cvssMetricV30']:
                cvss_score = metrics['cvssMetricV30'][0]['cvssData']['baseScore']
            
            year_cves.append({
                'cve_id': cve_id,
                'description': description,
                'cvss_score': cvss_score,
                'published_date': published_date,
            })

    except Exception as e:
        print(f"Warning: Problème lors du traitement de l'année {year}: {e}")
        
    return year_cves

def get_cisa_kev_data():
    """Télécharge le catalogue CISA KEV et crée un DataFrame pour la fusion."""
    try:
        kev_df = pd.read_csv(CISA_KEV_URL)
        kev_df = kev_df.rename(columns={'cveID': 'cve_id', 'dateAdded': 'kev_dateAdded'})
        kev_df['is_kev'] = 1 
        return kev_df[['cve_id', 'is_kev', 'kev_dateAdded', 'vendorProject', 
                       'vulnerabilityName', 'shortDescription', 'knownRansomwareCampaignUse']]
    except Exception as e:
        print(f"Erreur critique lors du téléchargement KEV : {e}")
        return pd.DataFrame()

def build_enriched_dataset():
    """
    Orchestre le téléchargement NVD, la fusion avec CISA KEV, et la sauvegarde 
    des datasets intermédiaires.
    """
    all_cves = []
    
    print(f"Initialisation de l'acquisition des données ({START_YEAR}-{END_YEAR})...")
    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Traitement NVD"):
        year_data = process_year_data(year)
        all_cves.extend(year_data)

    df_nvd = pd.DataFrame(all_cves)
    kev_df = get_cisa_kev_data()
    
    # Fusion (Left Join sur NVD pour garder toutes les vulnérabilités)
    df_nvd['is_kev'] = 0
    if not kev_df.empty:
        df_merged = df_nvd.merge(kev_df, on='cve_id', how='left', suffixes=('_nvd', '_kev'))
        df_merged['is_kev'] = df_merged['is_kev_kev'].fillna(0).astype(int)
        
        # Consolidation et remplissage
        df_merged['dateAdded'] = df_merged['kev_dateAdded'].fillna(df_merged['published_date'])
        df_merged['shortDescription'] = df_merged['shortDescription'].fillna(df_merged['description'])
        df_merged['vendorProject'] = df_merged['vendorProject'].fillna('Unknown')
        df_merged['vulnerabilityName'] = df_merged['vulnerabilityName'].fillna('Unknown')
        df_merged['knownRansomwareCampaignUse'] = df_merged['knownRansomwareCampaignUse'].fillna('Unknown')
        
        # Sélection des colonnes harmonisées
        final_cols = ['cve_id', 'dateAdded', 'cvss_score', 'is_kev', 
                      'vendorProject', 'vulnerabilityName', 'shortDescription', 
                      'knownRansomwareCampaignUse']
        df_final = df_merged[final_cols].copy()
    else:
        df_final = df_nvd.copy()

    # Sauvegarde des artefacts pour le module ML
    print("\nSauvegarde des datasets intermédiaires 'dataset_patched.csv' et 'dataset_unpatched.csv'...")
    df_final[df_final['is_kev'] == 1].to_csv('dataset_patched.csv', index=False)
    df_final[df_final['is_kev'] == 0].to_csv('dataset_unpatched.csv', index=False)
    
    return df_final

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    # 1. Chargement et Enrichissement des données
    print("=== DÉMARRAGE DU PIPELINE DE DONNÉES ===")
    df = build_enriched_dataset()
    
    # Conversion date pour traitement
    df['dateAdded'] = pd.to_datetime(df['dateAdded'], errors='coerce', utc=True)
    df = df.dropna(subset=['dateAdded']).reset_index(drop=True)

    print("\n=== APERÇU DU DATASET GLOBAL (ENRICHI) ===")
    print(f"Dimensions: {df.shape}")
    print(f"Période: {df['dateAdded'].min().date()} au {df['dateAdded'].max().date()}")
    print(f"Vulnérabilités exploitées (KEV): {df['is_kev'].sum()}")
    print("\n")
    print(df.head())
    
    # 2. Feature Engineering
    print("\n=== TRAITEMENT ET CRÉATION DES FEATURES ===")
    df_processed = df.copy()
    
    # Features temporelles
    df_processed['year_added'] = df_processed['dateAdded'].dt.year
    df_processed['month_added'] = df_processed['dateAdded'].dt.month
    
    # Indicateurs de sévérité basés sur les mots-clés
    def get_severity_label(name):
        name_lower = str(name).lower()
        if any(x in name_lower for x in ['remote code execution', 'code injection']):
            return 'Critical'
        elif any(x in name_lower for x in ['privilege escalation', 'sql injection']):
            return 'High'
        return 'Medium' if 'denial of service' in name_lower else 'Low'

    df_processed['severity_indicator'] = df_processed['vulnerabilityName'].apply(get_severity_label)
    
    # Indicateurs Ransomware
    df_processed['ransomware_target'] = (df_processed['knownRansomwareCampaignUse'] == 'Known').astype(int)

    # Statistiques après Feature Engineering
    print(f"\nDimensions du jeu de données final: {df_processed.shape}")
    print(f"Nouvelles colonnes créées: {list(set(df_processed.columns) - set(df.columns))}")
    
    print("\nPipeline de données terminé. Prêt pour la modélisation.")
