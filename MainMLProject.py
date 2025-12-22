"""
=============================================================================
MACHINE LEARNING PROJECT: VULNERABILITY RISK SCORING (CVE/KEV)
=============================================================================
Authors: Brayan KAMSU, Louis ESCUDIE, Lisa LANGLET
Date: December 12, 2025

OBJECTIVE:
Predict the likelihood of Common Vulnerabilities and Exposures (CVEs) being 
exploited based on historical evidence from NVD and CISA catalogs.
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import json
import io
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from tqdm import tqdm

# Scikit-Learn Ecosystem
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Advanced Gradient Boosting Model
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')

# 0. STRATEGIC CONFIGURATION 

class CFG:
    use_lightgbm_if_available = True
    start_year = 2002
    end_year = datetime.now().year
    nvd_feed_url = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{}.json.zip"
    cisa_kev_url = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
    ml_target_col = "is_kev"
    ml_date_col = "published_date"
    ml_test_size = 0.2
    ml_seed = 42

CVSS_METRICS_KEYS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# 1. DATA INGESTION (WEB ONLY) 

def load_data_from_web_only():
    print("\n[ETL] STARTING DATA EXTRACTION (WEB ONLY)...")
    
    # 1.1 Fetch CISA KEV (Targets) 
    try:
        kev_df = pd.read_csv(CFG.cisa_kev_url)
        kev_set = set(kev_df['cveID'].unique())
        print(f"    * CISA KEV Loaded: {len(kev_set)} entries.")
    except Exception as e:
        print(f"    ! Error fetching CISA KEV: {e}")
        kev_set = set()

    # 1.2 Fetch NVD Feeds (Features and Real Descriptions) 
    all_data = []
    for year in tqdm(range(CFG.start_year, CFG.end_year + 1), desc="Downloading Years"):
        try:
            r = requests.get(CFG.nvd_feed_url.format(year), timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    json_filename = z.namelist()[0]
                    data = json.loads(z.read(json_filename))
                    for item in data.get('vulnerabilities', []):
                        cve = item.get('cve', {})
                        metrics = cve.get('metrics', {})
                        
                        # Parsing CVSS Data
                        score, vector = np.nan, None
                        if 'cvssMetricV31' in metrics:
                            d = metrics['cvssMetricV31'][0]['cvssData']
                            score, vector = d.get('baseScore'), d.get('vectorString')
                        elif 'cvssMetricV30' in metrics:
                            d = metrics['cvssMetricV30'][0]['cvssData']
                            score, vector = d.get('baseScore'), d.get('vectorString')
                        elif 'cvssMetricV2' in metrics:
                            d = metrics['cvssMetricV2'][0]['cvssData']
                            score, vector = d.get('baseScore'), d.get('vectorString')
                        
                        # Extracting REAL English Description 
                        desc = next((d['value'] for d in cve.get('descriptions', []) if d['lang'] == 'en'), "")
                        
                        all_data.append({
                            'cve_id': cve.get('id'),
                            'published_date': cve.get('published'),
                            'description': desc,
                            'cvss_score': score,
                            'cvss_vector': vector
                        })
        except Exception:
            continue

    if not all_data:
        raise ValueError("CRITICAL: No data could be downloaded.")
        
    df = pd.DataFrame(all_data)
    # Mapping the target label (KEV vs Non-KEV) 
    df[CFG.ml_target_col] = df['cve_id'].apply(lambda x: 1 if x in kev_set else 0)
    print(f"    * Consolidated dataset contains {len(df)} records.")
    return df

# 2. DATA PRE-PROCESSING 

def preprocess_data(df):
    print("\n[STEP 2] PRE-PROCESSING & FEATURE ENGINEERING")
    
    # Cleaning duplicates 
    df = df.copy().drop_duplicates(subset=['cve_id'])
    
    # Temporal Information 
    df[CFG.ml_date_col] = pd.to_datetime(df[CFG.ml_date_col], errors='coerce', utc=True)
    df = df.dropna(subset=[CFG.ml_date_col])
    now = pd.Timestamp.now(tz='UTC')
    df['age_days'] = (now - df[CFG.ml_date_col]).dt.days
    df['desc_len'] = df['description'].fillna("").str.len()
    
    # CVSS Vector Parsing
    def parse_vector(v):
        out = {m: "MISSING" for m in CVSS_METRICS_KEYS}
        if isinstance(v, str):
            for part in v.split('/'):
                if ':' in part:
                    k, val = part.split(':', 1)
                    if k in out: out[k] = val
        return out

    cvss_feats = df['cvss_vector'].apply(parse_vector).apply(pd.Series)
    df = pd.concat([df, cvss_feats], axis=1)
    
    return df

# 3. MODELLING STRATEGY & IMPLEMENTATION

def build_model():
    """Implements the decision logic to pivot to LightGBM[cite: 156, 161]."""
    if CFG.use_lightgbm_if_available and HAS_LIGHTGBM:
        print("-> Selecting LightGBM (Gradient Boosting)")
        return LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=CFG.ml_seed,
            verbose=-1,
            scale_pos_weight=10 # Handling severe imbalance 
        )
    else:
        print("-> Fallback to Logistic Regression")
        return LogisticRegression(max_iter=1000, class_weight='balanced')

def train_production_model(df):
    print("\n[STEP 3] MODELLING & TUNING")
    
    num_cols = ['cvss_score', 'age_days', 'desc_len']
    cat_cols = CVSS_METRICS_KEYS
    
    # Pipeline construction 
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)) # Strategic dimension reduction
    ])
    
    preprocessor = ColumnTransformer([
        ('txt', TfidfVectorizer(max_features=5000, stop_words='english'), 'description'),
        ('num', num_pipe, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Temporal Split 
    df = df.sort_values(CFG.ml_date_col)
    limit = int(len(df) * (1 - CFG.ml_test_size))
    train_df, test_df = df.iloc[:limit], df.iloc[limit:]
    
    X_train = train_df.drop(columns=[CFG.ml_target_col, 'cve_id', 'published_date', 'cvss_vector'])
    y_train = train_df[CFG.ml_target_col]
    
    model_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', build_model())
    ])

    # Extended Training on the Full Dataset 
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

# 4. STRATEGIC INTERPRETATION & RESULTS 

def generate_risk_stratification(model, df):
    print("\n[STEP 4] RISK-BASED STRATIFICATION [cite: 207]")
    
    # Focusing on currently unexploited CVEs (Non-KEV) 
    non_kev = df[df[CFG.ml_target_col] == 0].copy()
    X_target = non_kev.drop(columns=[CFG.ml_target_col, 'cve_id', 'published_date', 'cvss_vector'])
    
    probs = model.predict_proba(X_target)[:, 1]
    non_kev['risk_score'] = probs
    
    # 4-Tier Quantile Stratification 
    q99, q95, q80 = np.quantile(probs, [0.99, 0.95, 0.80])
    
    def get_tier(p):
        if p >= q99: return "Critical (top 1%)"
        if p >= q95: return "High (1-5%)"
        if p >= q80: return "Medium (5-20%)"
        return "Low (rest)"
    
    non_kev['risk_tier'] = non_kev['risk_score'].apply(get_tier)
    
    # Exporting result with the REAL descriptions as requested
    output_cols = ['cve_id', 'risk_score', 'risk_tier', 'cvss_score', 'description']
    non_kev[output_cols].to_csv("non_kve_predicted_probs.csv", index=False)
    
    print("Risk Tier Distribution Summary[cite: 231]:")
    print(non_kev['risk_tier'].value_counts().sort_index(ascending=False))

if __name__ == "__main__":
    # 1. Real Data Extraction (Web Only)
    data = load_data_from_web_only()
    
    # 2. Advanced Pre-processing
    processed_data = preprocess_data(data)
    
    # 3. LightGBM Pipeline Training
    final_pipeline = train_production_model(processed_data)
    
    # 4. Final Stratification & CSV Generation
    generate_risk_stratification(final_pipeline, processed_data)
    
    print("\nProcessing Complete. File 'non_kve_predicted_probs.csv' generated.")
    
    
