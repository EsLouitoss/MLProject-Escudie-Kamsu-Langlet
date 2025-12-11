import pandas as pd
import numpy as np
import requests
import zipfile
import json
import io
import re
import warnings
import os
from datetime import datetime, timezone
from tqdm import tqdm
from pathlib import Path

# --- Imports Machine Learning ---
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Gestion LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Info : LightGBM non détecté. Le script basculera sur RandomForest.")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
START_YEAR = 2002
END_YEAR = datetime.now().year
NVD_FEED_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{}.json.zip"
CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"

ML_TARGET_COL = "is_kev"
ML_DATE_COL = "published_date"
ML_TEST_SIZE = 0.2
ML_SEED = 42

KEYWORD_PATTERNS = {
    "has_rce": r"\b(remote code execution|rce|code execution)\b",
    "has_auth_bypass": r"\b(auth(entication)? bypass|bypass auth|authorization bypass)\b",
    "has_deserialization": r"\b(deserialization|unsafe deserialization)\b",
    "has_sqli": r"\b(sql injection|sqli)\b",
    "has_xss": r"\b(cross[-\s]?site scripting|xss)\b",
    "has_path_traversal": r"\b(path traversal|directory traversal)\b",
    "has_privesc": r"\b(privilege escalation|privesc|elevation of privilege)\b",
}

CVSS_METRICS_KEYS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# ---------------------------------------------------------
# 1. Gestion des données (ETL)
# ---------------------------------------------------------

def parse_cvss_vector(vector_str):
    out = {m: "MISSING" for m in CVSS_METRICS_KEYS}
    if not isinstance(vector_str, str):
        return out
    parts = vector_str.split('/')
    for part in parts:
        if ':' in part:
            key, val = part.split(':', 1)
            if key in out:
                out[key] = val
    return out

def try_load_local_files():
    """Charge les CSV locaux s'ils existent."""
    print("Vérification des fichiers locaux...")
    files_csv = ('dataset_patched.csv', 'dataset_unpatched.csv')
    
    if os.path.exists(files_csv[0]) and os.path.exists(files_csv[1]):
        try:
            df_patched = pd.read_csv(files_csv[0])
            df_unpatched = pd.read_csv(files_csv[1])
            
            df_patched[ML_TARGET_COL] = 1
            df_unpatched[ML_TARGET_COL] = 0
            
            df_full = pd.concat([df_patched, df_unpatched], ignore_index=True)
            print(f" -> Chargement local réussi : {len(df_full)} entrées.")
            return df_full
        except Exception as e:
            print(f" -> Erreur lecture CSV: {e}")
    
    print(" -> Fichiers locaux absents ou incomplets. Passage au téléchargement.")
    return None

def process_year_data(year):
    """Téléchargement NVD."""
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
                
        for item in data.get('vulnerabilities', []):
            cve = item.get('cve', {})
            cve_id = cve.get('id')
            
            desc_text = ""
            for d in cve.get('descriptions', []):
                if d['lang'] == 'en':
                    desc_text = d['value']
                    break
            
            published = cve.get('published')
            
            score = np.nan
            vector = None
            metrics = cve.get('metrics', {})
            
            if 'cvssMetricV31' in metrics:
                data = metrics['cvssMetricV31'][0]['cvssData']
                score, vector = data.get('baseScore'), data.get('vectorString')
            elif 'cvssMetricV30' in metrics:
                data = metrics['cvssMetricV30'][0]['cvssData']
                score, vector = data.get('baseScore'), data.get('vectorString')
            elif 'cvssMetricV2' in metrics:
                data = metrics['cvssMetricV2'][0]['cvssData']
                score, vector = data.get('baseScore'), data.get('vectorString') 

            has_patch_ref = 0
            for ref in cve.get('references', []):
                tags = ref.get('tags', [])
                if tags and 'Patch' in tags:
                    has_patch_ref = 1
                    break

            year_cves.append({
                'cve_id': cve_id,
                'published_date': published,
                'description': desc_text,
                'cvss_score': score,
                'cvss_vector': vector,
                'has_patch_ref': has_patch_ref
            })

    except Exception as e:
        print(f"Erreur NVD année {year}: {e}")
        
    return year_cves

def get_cisa_kev_data():
    print("Tentative de téléchargement CISA KEV...")
    try:
        df = pd.read_csv(CISA_KEV_URL)
        return set(df['cveID'].unique())
    except Exception as e:
        print(f"Erreur CISA Web ({e}).")
        return set()

def build_dataset():
    # 1. Essai Local
    df = try_load_local_files()
    
    # 2. Essai Web
    if df is None:
        print(f"Démarrage téléchargement NVD ({START_YEAR}-{END_YEAR})")
        all_data = []
        for year in tqdm(range(START_YEAR, END_YEAR + 1)):
            all_data.extend(process_year_data(year))
        
        df = pd.DataFrame(all_data)
        if df.empty:
            raise ValueError("Impossible de récupérer des données.")
            
        kev_set = get_cisa_kev_data()
        df[ML_TARGET_COL] = df['cve_id'].apply(lambda x: 1 if x in kev_set else 0)
        
        print("Sauvegarde des données...")
        df[df[ML_TARGET_COL] == 1].to_csv('dataset_patched.csv', index=False)
        df[df[ML_TARGET_COL] == 0].to_csv('dataset_unpatched.csv', index=False)
    
    # 3. Nettoyage
    df[ML_DATE_COL] = pd.to_datetime(df[ML_DATE_COL], errors='coerce', utc=True)
    df = df.dropna(subset=[ML_DATE_COL]).sort_values(ML_DATE_COL).reset_index(drop=True)
    
    if 'published' in df.columns and 'published_date' not in df.columns:
        df.rename(columns={'published': 'published_date'}, inplace=True)
    if 'shortDescription' in df.columns and 'description' not in df.columns:
        df.rename(columns={'shortDescription': 'description'}, inplace=True)
        
    print(f"Dataset prêt : {df.shape} | Exploités (KEV) : {df[ML_TARGET_COL].sum()}")
    return df

# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------

def feature_engineering(df):
    df = df.copy()
    print("Génération des features...")

    desc = df['description'].fillna("").astype(str)
    
    for col, pat in KEYWORD_PATTERNS.items():
        df[col] = desc.str.contains(pat, flags=re.IGNORECASE, regex=True).astype(int)
    
    if 'cvss_vector' in df.columns:
        df['cvss_vector'] = df['cvss_vector'].fillna("")
        cvss_parsed = df['cvss_vector'].apply(parse_cvss_vector).apply(pd.Series)
        df = pd.concat([df, cvss_parsed], axis=1)

    now = pd.Timestamp(datetime.now(timezone.utc))
    df['age_days'] = (now - df[ML_DATE_COL]).dt.days
    df['desc_len'] = desc.str.len()
    
    if 'has_patch_ref' not in df.columns:
        df['has_patch_ref'] = 0
    
    return df

# ---------------------------------------------------------
# 3. Machine Learning (Undersampling + LightGBM)
# ---------------------------------------------------------

def train_evaluate_model(df):
    if df[ML_TARGET_COL].nunique() < 2:
        print("Erreur : Une seule classe détectée.")
        return None

    # Split Temporel
    limit_idx = int(len(df) * (1 - ML_TEST_SIZE))
    train_df = df.iloc[:limit_idx]
    test_df = df.iloc[limit_idx:]
    
    print(f"Données Train Originales : {len(train_df)}")

    # Undersampling (Ratio 1:20)
    kev_train = train_df[train_df[ML_TARGET_COL] == 1]
    safe_train = train_df[train_df[ML_TARGET_COL] == 0]
    
    if len(kev_train) > 0:
        n_safe = len(kev_train) * 20
        safe_train_sampled = safe_train.sample(n=n_safe, random_state=ML_SEED)
        train_df_balanced = pd.concat([kev_train, safe_train_sampled]).sample(frac=1, random_state=ML_SEED)
    else:
        print("Attention: Pas de KEV dans le set d'entraînement.")
        train_df_balanced = train_df

    print(f"Données Train Rééquilibrées : {len(train_df_balanced)} (dont {len(kev_train)} KEV)")
    
    cols_drop = [ML_TARGET_COL, 'cve_id', 'published_date', 'cvss_vector', 'published']
    X_train = train_df_balanced.drop(columns=cols_drop, errors='ignore')
    y_train = train_df_balanced[ML_TARGET_COL]
    
    X_test = test_df.drop(columns=cols_drop, errors='ignore')
    y_test = test_df[ML_TARGET_COL]
    
    avail_cols = X_train.columns.tolist()
    cat_cols = [c for c in CVSS_METRICS_KEYS if c in avail_cols]
    num_cols = ['cvss_score', 'age_days', 'desc_len', 'has_patch_ref'] + list(KEYWORD_PATTERNS.keys())
    num_cols = [c for c in num_cols if c in avail_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('txt', TfidfVectorizer(max_features=2000, stop_words='english'), 'description'),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    
    if HAS_LIGHTGBM:
        clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, scale_pos_weight=1.5, n_jobs=-1, random_state=ML_SEED, verbose=-1)
    else:
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=ML_SEED)
        
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    print("Entraînement en cours...")
    pipeline.fit(X_train, y_train)
    
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    print("\n--- RÉSULTATS (Test Set) ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"Avg Precision: {average_precision_score(y_test, probs):.4f}")
    print(classification_report(y_test, preds))
    
    return pipeline

def generate_risk_report(pipeline, df):
    if pipeline is None: return

    print("\nPrédiction sur les CVE non-classées (Risk Scoring)...")
    target_df = df[df[ML_TARGET_COL] == 0].copy()
    
    if target_df.empty:
        return

    cols_drop = [ML_TARGET_COL, 'cve_id', 'published_date', 'cvss_vector', 'published']
    X_target = target_df.drop(columns=cols_drop, errors='ignore')
    
    target_df['risk_probability'] = pipeline.predict_proba(X_target)[:, 1]
    
    q99 = target_df['risk_probability'].quantile(0.99)
    q95 = target_df['risk_probability'].quantile(0.95)
    
    def get_tier(p):
        if p >= q99: return "CRITICAL"
        if p >= q95: return "HIGH"
        return "MEDIUM/LOW"
    
    target_df['risk_tier'] = target_df['risk_probability'].apply(get_tier)
    
    out_cols = ['cve_id', 'risk_probability', 'risk_tier', 'cvss_score', 'description']
    final_df = target_df[out_cols].sort_values('risk_probability', ascending=False)
    
    final_df.to_csv("predicted_risk_report.csv", index=False)
    print("Rapport 'predicted_risk_report.csv' généré avec succès.")
    print("Top 3 CVE détectées à haut risque :")
    print(final_df[['cve_id', 'risk_probability', 'description']].head(3))

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    df_raw = build_dataset()
    df_ml = feature_engineering(df_raw)
    model = train_evaluate_model(df_ml)
    generate_risk_report(model, df_ml)
