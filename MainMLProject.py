import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import warnings
import requests
import zipfile
import json
import io
import re
from tqdm import tqdm

# Imports Machine Learning (issus de Lisa.py)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Tentative d'import LightGBM (optionnel)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')

# Configuration du style
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

# Configuration ML
ML_TARGET_COL = "is_kev"
ML_TEXT_COL = "shortDescription"
ML_DATE_COL = "dateAdded"
ML_TEST_SIZE = 0.2
ML_SEED = 42

# Mots-clés pour le Feature Engineering (Lisa.py)
KEYWORD_PATTERNS = {
    "has_rce": r"\b(remote code execution|rce|code execution)\b",
    "has_auth_bypass": r"\b(auth(entication)? bypass|bypass auth|authorization bypass)\b",
    "has_deserialization": r"\b(deserialization|unsafe deserialization)\b",
    "has_sqli": r"\b(sql injection|sqli)\b",
    "has_xss": r"\b(cross[-\s]?site scripting|xss)\b",
    "has_path_traversal": r"\b(path traversal|directory traversal)\b",
    "has_privesc": r"\b(privilege escalation|privesc|elevation of privilege)\b",
}

# ==============================================================================
# --- PARTIE 1 : FONCTIONS D'ACQUISITION DE DONNÉES (ETL) ---
# ==============================================================================

def process_year_data(year):
    """Télécharge et extrait les données NVD pour une année."""
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
    """Télécharge le catalogue CISA KEV."""
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
    """Orchestre l'ETL complet et sauvegarde les fichiers intermédiaires."""
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

    print("\nSauvegarde des datasets intermédiaires...")
    # Sauvegarde des artefacts pour le module ML
    print("\nSauvegarde des datasets intermédiaires 'dataset_patched.csv' et 'dataset_unpatched.csv'...")
    df_final[df_final['is_kev'] == 1].to_csv('dataset_patched.csv', index=False)
    df_final[df_final['is_kev'] == 0].to_csv('dataset_unpatched.csv', index=False)
    
    return df_final

# ==============================================================================
# --- PARTIE 2 : FONCTIONS MACHINE LEARNING (Issues de Lisa.py) ---
# ==============================================================================

def add_ml_features(df):
    """Pipeline de feature engineering spécifique pour le ML."""
    df = df.copy()
    
    # 1. NLP Flags (Regex)
    desc = df[ML_TEXT_COL].fillna("").astype(str)
    for col, pat in KEYWORD_PATTERNS.items():
        df[col] = desc.str.contains(pat, flags=re.IGNORECASE, regex=True).astype(int)
        
    # 2. Features Temporelles Avancées
    today = pd.Timestamp(datetime.now(timezone.utc))
    # Conversion UTC explicite pour éviter les erreurs de comparaison
    if df[ML_DATE_COL].dt.tz is None:
        df[ML_DATE_COL] = df[ML_DATE_COL].dt.tz_localize('UTC')
    else:
        df[ML_DATE_COL] = df[ML_DATE_COL].dt.tz_convert('UTC')
        
    df["age_days"] = (today - df[ML_DATE_COL]).dt.days
    df["desc_len"] = desc.str.len()
    
    return df

def temporal_train_test_split(df, date_col, test_frac):
    """Split temporel strict (Train = Passé, Test = Futur)."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n_test = int(len(df) * test_frac)
    return df.iloc[:-n_test], df.iloc[-n_test:]

def build_ml_pipeline(numeric_cols):
    """Construit le pipeline Scikit-Learn (Preprocessing + Modèle)."""
    
    # Choix du modèle
    if HAS_LIGHTGBM:
        print(" -> Utilisation du modèle LightGBM (Optimisé).")
        clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=ML_SEED, n_jobs=-1, verbose=-1)
    else:
        print(" -> Utilisation du modèle RandomForest (Standard).")
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=ML_SEED, n_jobs=-1)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2)), ML_TEXT_COL),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
        ],
        remainder="drop"
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    # --- PHASE 1 : ACQUISITION DE DONNÉES ---
    print("=== ÉTAPE 1/3 : DÉMARRAGE DU PIPELINE DE DONNÉES (ETL) ===")
    df = build_enriched_dataset()
    
    # Nettoyage initial
    df['dateAdded'] = pd.to_datetime(df['dateAdded'], errors='coerce', utc=True)
    df = df.dropna(subset=['dateAdded']).reset_index(drop=True)

    print("\n=== APERÇU DU DATASET ===")
    print(f"Dimensions: {df.shape}")
    print(f"Exploités (KEV): {df['is_kev'].sum()} | Non-exploités: {(df['is_kev']==0).sum()}")
    
    # --- PHASE 2 : FEATURE ENGINEERING & ANALYSE SIMPLE ---
    print("\n=== ÉTAPE 2/3 : PRÉPARATION ET ANALYSE RAPIDE ===")
    df_processed = df.copy()
    
    # Indicateurs basiques
    df_processed['year_added'] = df_processed['dateAdded'].dt.year
    df_processed['severity_indicator'] = df_processed['vulnerabilityName'].apply(
        lambda x: 'Critical' if 'code execution' in str(x).lower() else 'Medium'
    )
    
    print("Analyse préliminaire terminée.")
    
    # --- PHASE 3 : MACHINE LEARNING (Intégration de Lisa.py) ---
    print("\n=== ÉTAPE 3/3 : MODÉLISATION PRÉDICTIVE (AI RISK SCORING) ===")
    
    # 3.1 Enrichissement ML
    print("Génération des features avancées (NLP, Regex, Age)...")
    df_ml = add_ml_features(df_processed)
    
    # Identification des colonnes numériques pour le modèle
    numeric_cols = ["cvss_score", "desc_len", "age_days"]
    numeric_cols += [k for k in KEYWORD_PATTERNS.keys() if k in df_ml.columns]
    
    # 3.2 Split Train/Test
    train_df, test_df = temporal_train_test_split(df_ml, ML_DATE_COL, ML_TEST_SIZE)
    print(f"Split Temporel -> Train: {len(train_df)} | Test: {len(test_df)}")
    
    # 3.3 Entraînement
    print("Entraînement du modèle en cours...")
    pipeline = build_ml_pipeline(numeric_cols)
    pipeline.fit(train_df, train_df[ML_TARGET_COL])
    
    # 3.4 Évaluation
    print("\n--- Évaluation du Modèle ---")
    proba_test = pipeline.predict_proba(test_df)[:, 1]
    y_test = test_df[ML_TARGET_COL]
    
    print(f"ROC-AUC Score : {roc_auc_score(y_test, proba_test):.4f}")
    print("Rapport de Classification (Top 3 lignes) :")
    print(classification_report(y_test, (proba_test >= 0.5).astype(int), digits=3))
    
    # 3.5 Prédiction de Risque (Sur les données NON KEV)
    print("\n--- Calcul des scores de risque pour les vulnérabilités non exploitées ---")
    non_kev_df = df_ml[df_ml[ML_TARGET_COL] == 0].copy()
    
    if not non_kev_df.empty:
        non_kev_df["risk_score"] = pipeline.predict_proba(non_kev_df)[:, 1]
        
        # Définition des tiers de risque
        q99 = non_kev_df["risk_score"].quantile(0.99)
        q95 = non_kev_df["risk_score"].quantile(0.95)
        
        def get_tier(score):
            if score >= q99: return "CRITICAL (Top 1%)"
            if score >= q95: return "HIGH (Top 5%)"
            return "LOW/MEDIUM"

        non_kev_df["risk_tier"] = non_kev_df["risk_score"].apply(get_tier)
        
        # Sauvegarde
        output_path = "predictions_risk_report.csv"
        cols_export = ["cve_id", "risk_score", "risk_tier", "cvss_score", "vulnerabilityName"]
        non_kev_df[cols_export].sort_values("risk_score", ascending=False).to_csv(output_path, index=False)
        
        print(f"Succès ! Rapport de risque sauvegardé sous : {output_path}")
        print("Top 3 CVEs à haut risque détectées :")
        print(non_kev_df[cols_export].sort_values("risk_score", ascending=False).head(3))
    else:
        print("Aucune donnée non-KEV à analyser.")

    print("\n=== PROGRAMME TERMINÉ AVEC SUCCÈS ===")
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
