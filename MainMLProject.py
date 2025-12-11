"""
=============================================================================
MACHINE LEARNING PROJECT: VULNERABILITY RISK SCORING (CVE/KEV)
=============================================================================
Student Name: [Your Name]
Date: December 2025

OBJECTIVE:
Predict whether a CVE (Common Vulnerabilities and Exposures) is likely to be 
exploited (KEV - Known Exploited Vulnerability) based on its description 
and metrics.

MODE: 
FULL WEB FETCH (No local cache reading) to ensure data integrity.
"""

# =============================================================================
# 1. IMPORTATION OF LIBRARIES AND DATASET
# =============================================================================
import pandas as pd
import numpy as np
import requests
import zipfile
import json
import io
import re
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from tqdm import tqdm
from pathlib import Path

# Scikit-Learn Ecosystem
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (roc_auc_score, classification_report, 
                             average_precision_score, confusion_matrix, 
                             f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Try importing LightGBM (Advanced Model)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Info: LightGBM not found. Skipping advanced gradient boosting models.")

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
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

# --- DATA LOADING FUNCTIONS ---
def parse_cvss_vector(vector_str):
    out = {m: "MISSING" for m in CVSS_METRICS_KEYS}
    if not isinstance(vector_str, str): return out
    parts = vector_str.split('/')
    for part in parts:
        if ':' in part:
            key, val = part.split(':', 1)
            if key in out: out[key] = val
    return out

def load_data_from_web_only():
    """
    FORCE DOWNLOAD MODE:
    Ignores local CSV files to prevent data corruption. 
    Always fetches fresh data from NVD and CISA.
    """
    print("\n[ETL] STARTING DATA EXTRACTION (WEB ONLY)...")
    
    # 1. Download Target Labels (CISA KEV)
    print(" -> Fetching CISA Known Exploited Vulnerabilities...")
    kev_set = set()
    try:
        kev_df = pd.read_csv(CISA_KEV_URL)
        kev_set = set(kev_df['cveID'].unique())
        print(f"    * CISA KEV Loaded: {len(kev_set)} entries.")
    except Exception as e:
        print(f"    ! Error fetching CISA KEV: {e}")

    # 2. Download Features (NVD Feeds)
    print(f" -> Fetching NVD Feeds ({START_YEAR}-{END_YEAR})...")
    all_data = []
    
    for year in tqdm(range(START_YEAR, END_YEAR + 1)):
        try:
            r = requests.get(NVD_FEED_URL.format(year), timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    # NVD JSON parsing
                    json_filename = z.namelist()[0]
                    data = json.loads(z.read(json_filename))
                    
                    for item in data.get('vulnerabilities', []):
                        cve = item.get('cve', {})
                        metrics = cve.get('metrics', {})
                        
                        # Prioritize V3.1 > V3.0 > V2
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
                        
                        # Extract description (English)
                        desc = next((d['value'] for d in cve.get('descriptions', []) if d['lang'] == 'en'), "")
                        
                        # Check for 'Patch' tag in references
                        has_patch = 0
                        for ref in cve.get('references', []):
                            if 'tags' in ref and 'Patch' in ref['tags']:
                                has_patch = 1; break
                        
                        all_data.append({
                            'cve_id': cve.get('id'),
                            'published_date': cve.get('published'),
                            'description': desc,
                            'cvss_score': score,
                            'cvss_vector': vector,
                            'has_patch_ref': has_patch
                        })
        except Exception as e:
            print(f"    ! Error processing year {year}: {e}")

    if not all_data:
        raise ValueError("CRITICAL: No data could be downloaded from NVD.")

    df = pd.DataFrame(all_data)
    
    # Labeling
    df[ML_TARGET_COL] = df['cve_id'].apply(lambda x: 1 if x in kev_set else 0)
    
    # Save a backup (just in case, but we won't read it next time)
    print(" -> Data extraction complete. Saving backup CSVs...")
    try:
        df[df[ML_TARGET_COL] == 1].to_csv('dataset_patched.csv', index=False)
        df[df[ML_TARGET_COL] == 0].to_csv('dataset_unpatched.csv', index=False)
    except:
        pass
    
    return df

# =============================================================================
# 2. DATASET ANALYSIS (Quality, Visu, Correlation)
# =============================================================================
def analyze_dataset(df):
    print("\n" + "="*50)
    print("STEP 2: DATASET ANALYSIS")
    print("="*50)
    
    # 2.1 Basic Stats
    print(f"Dimensions: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 2.2 Data Balance
    counts = df[ML_TARGET_COL].value_counts()
    print(f"\nTarget Distribution ({ML_TARGET_COL}):\n{counts}")
    ratio = counts[1] / (counts[0] + counts[1]) * 100
    print(f"Positive Class Ratio: {ratio:.2f}% (High Imbalance!)")
    
    # 2.3 Visualization
    try:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=ML_TARGET_COL, data=df, palette='viridis')
        plt.title('Target Distribution (0=Safe, 1=Exploited)')
        plt.yscale('log')
        plt.ylabel('Count (Log Scale)')
        plt.show()
    except Exception:
        pass

# =============================================================================
# 3. DATA PRE-PROCESSING
# =============================================================================
def preprocess_data(df):
    print("\n" + "="*50)
    print("STEP 3: DATA PRE-PROCESSING")
    print("="*50)
    
    df = df.copy()
    
    # 3.1 Cleaning
    init_len = len(df)
    df = df.drop_duplicates(subset=['cve_id'])
    
    # Handle Dates (Crucial Step)
    df[ML_DATE_COL] = pd.to_datetime(df[ML_DATE_COL], errors='coerce', utc=True)
    df = df.dropna(subset=[ML_DATE_COL])
    
    print(f"Duplicates/Invalid Dates removed: {init_len - len(df)}")
    
    # 3.2 Feature Engineering
    print("-> Creating features from text and dates...")
    
    # NLP / Regex Features
    desc = df['description'].fillna("").astype(str)
    for col, pat in KEYWORD_PATTERNS.items():
        df[col] = desc.str.contains(pat, flags=re.IGNORECASE, regex=True).astype(int)
    
    # Temporal Features
    now = pd.Timestamp(datetime.now(timezone.utc))
    df['age_days'] = (now - df[ML_DATE_COL]).dt.days
    df['desc_len'] = desc.str.len()
    
    # CVSS Vector Parsing
    if 'cvss_vector' in df.columns:
        df['cvss_vector'] = df['cvss_vector'].fillna("")
        cvss_parsed = df['cvss_vector'].apply(parse_cvss_vector).apply(pd.Series)
        df = pd.concat([df, cvss_parsed], axis=1)

    return df

# =============================================================================
# 4. DIMENSION REDUCTION (PCA)
# =============================================================================
def perform_pca_analysis(X_train_num):
    print("\n" + "="*50)
    print("STEP 4: DIMENSION REDUCTION (PCA)")
    print("="*50)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_num)
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original Num Features: {X_train_num.shape[1]}")
    print(f"Reduced Components (95% var): {X_pca.shape[1]}")

# =============================================================================
# 5. MODELING (Creation, Tuning, Addressing Overfitting)
# =============================================================================
def build_and_train_models(df):
    print("\n" + "="*50)
    print("STEP 5: MODELING & TUNING")
    print("="*50)
    
    # 5.1 Train-Test Split (Temporal)
    df = df.sort_values(ML_DATE_COL)
    limit = int(len(df) * (1 - ML_TEST_SIZE))
    
    train_df = df.iloc[:limit]
    test_df = df.iloc[limit:]
    
    # 5.2 Addressing Class Imbalance (Undersampling)
    print("-> Addressing Class Imbalance (Undersampling Strategy)...")
    kev_train = train_df[train_df[ML_TARGET_COL] == 1]
    safe_train = train_df[train_df[ML_TARGET_COL] == 0]
    
    # Ratio 1:20
    if len(kev_train) > 0:
        safe_sample = safe_train.sample(n=len(kev_train)*20, random_state=ML_SEED)
        train_balanced = pd.concat([kev_train, safe_sample]).sample(frac=1, random_state=ML_SEED)
    else:
        print("! WARNING: No KEV in training set.")
        train_balanced = train_df
        
    print(f"Train Set Size: {len(train_balanced)} (Balanced)")
    print(f"Test Set Size: {len(test_df)} (Real World Distribution)")

    # Prepare X, y
    cols_drop = [ML_TARGET_COL, 'cve_id', 'published_date', 'cvss_vector', 'published']
    X_train = train_balanced.drop(columns=cols_drop, errors='ignore')
    y_train = train_balanced[ML_TARGET_COL]
    X_test = test_df.drop(columns=cols_drop, errors='ignore')
    y_test = test_df[ML_TARGET_COL]

    # Feature selection for Preprocessor
    avail_cols = X_train.columns.tolist()
    cat_cols = [c for c in CVSS_METRICS_KEYS if c in avail_cols]
    num_cols = ['cvss_score', 'age_days', 'desc_len', 'has_patch_ref'] + list(KEYWORD_PATTERNS.keys())
    num_cols = [c for c in num_cols if c in avail_cols]

    # Run PCA Analysis demo
    perform_pca_analysis(X_train[num_cols].fillna(0))

    # 5.3 Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('txt', TfidfVectorizer(max_features=1000, stop_words='english'), 'description'),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    results = {}
    
    # MODEL 1: Logistic Regression
    print("\nTraining Model 1: Logistic Regression (Baseline)...")
    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    pipe_lr.fit(X_train, y_train)
    results['Logistic Regression'] = pipe_lr

    # MODEL 2: Random Forest
    print("Training Model 2: Random Forest (Ensemble)...")
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                       class_weight='balanced', random_state=ML_SEED, n_jobs=-1))
    ])
    pipe_rf.fit(X_train, y_train)
    results['Random Forest'] = pipe_rf

    # MODEL 3: LightGBM
    if HAS_LIGHTGBM:
        print("Training Model 3: LightGBM (Gradient Boosting)...")
        # Scale_pos_weight helps even with undersampling
        pipe_lgbm = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LGBMClassifier(n_estimators=1000, learning_rate=0.05, 
                                   num_leaves=31, scale_pos_weight=1.5, 
                                   verbose=-1, random_state=ML_SEED))
        ])
        pipe_lgbm.fit(X_train, y_train)
        results['LightGBM'] = pipe_lgbm

    return results, X_test, y_test

# =============================================================================
# 6. MODEL COMPARISON & CONCLUSION
# =============================================================================
def compare_models(results, X_test, y_test):
    print("\n" + "="*50)
    print("STEP 6: MODEL COMPARISON")
    print("="*50)
    
    summary = []
    
    for name, model in results.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        # Calculate Recall for Class 1 specifically
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        recall_pos = tp / (tp + fn) if (tp+fn) > 0 else 0
        
        summary.append({
            'Model': name,
            'ROC-AUC': auc,
            'Avg Precision': ap,
            'F1-Score': f1,
            'Recall (Class 1)': recall_pos
        })
        
    summary_df = pd.DataFrame(summary)
    print(summary_df.sort_values(by='ROC-AUC', ascending=False))
    
    # Pick Best
    best_model_name = summary_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]['Model']
    print(f"\nCONCLUSION: The best performing model is {best_model_name}.")
    
    return results[best_model_name]

# =============================================================================
# 7. BONUS: RISK SCORING REPORT
# =============================================================================
def generate_risk_report(model, df):
    print("\n" + "="*50)
    print("STEP 7: RISK SCORING REPORT (BONUS)")
    print("="*50)
    
    # Predict on safe CVEs
    target_df = df[df[ML_TARGET_COL] == 0].copy()
    if target_df.empty: return
    
    cols_drop = [ML_TARGET_COL, 'cve_id', 'published_date', 'cvss_vector', 'published']
    X_target = target_df.drop(columns=cols_drop, errors='ignore')
    
    probs = model.predict_proba(X_target)[:, 1]
    target_df['risk_score'] = probs
    
    # Tiering
    q99 = np.quantile(probs, 0.99)
    q95 = np.quantile(probs, 0.95)
    
    def get_tier(p):
        if p >= q99: return "CRITICAL"
        if p >= q95: return "HIGH"
        return "MEDIUM/LOW"
        
    target_df['risk_tier'] = target_df['risk_score'].apply(get_tier)
    
    # Export
    output_cols = ['cve_id', 'risk_score', 'risk_tier', 'cvss_score', 'description']
    report = target_df[output_cols].sort_values('risk_score', ascending=False).head(50)
    report.to_csv("final_risk_report.csv", index=False)
    
    print("Top 5 Detected Risks:")
    print(report.head(5))
    print("\nFile 'final_risk_report.csv' saved successfully.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Loading (Force Web)
    df = load_data_from_web_only()
    
    # 2. EDA
    analyze_dataset(df)
    
    # 3. Preprocessing
    df_clean = preprocess_data(df)
    
    # 4, 5, 6. Modeling Pipeline
    models, X_test, y_test = build_and_train_models(df_clean)
    best_model = compare_models(models, X_test, y_test)
    
    # 7. Production
    generate_risk_report(best_model, df_clean)
