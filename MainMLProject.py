import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('known_exploited_vulnerabilities.csv')

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Basic dataset information
print("=== DATASET OVERVIEW ===")
print(f"Number of vulnerabilities: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print(f"Date range: {df['dateAdded'].min()} to {df['dateAdded'].max()}")

# Display basic info
print("\n=== DATASET INFO ===")
print(df.info())

# Display first few rows
print("\n=== FIRST 5 ROWS ===")
display(df.head())

# Check for missing values
print("\n=== MISSING VALUES ===")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_info[missing_info['Missing Count'] > 0])

# Basic statistics for numerical columns
print("\n=== BASIC STATISTICS ===")
print(df.describe())

# Key metrics analysis
print("=== KEY METRICS ===")
print(f"Unique vendors: {df['vendorProject'].nunique()}")
print(f"Unique products: {df['product'].nunique()}")
print(f"Vulnerabilities with known ransomware use: {df['knownRansomwareCampaignUse'].value_counts().get('Known', 0)}")
print(f"Vulnerabilities with unknown ransomware use: {df['knownRansomwareCampaignUse'].value_counts().get('Unknown', 0)}")

# Vendor distribution
print("\n=== TOP 10 VENDORS BY VULNERABILITY COUNT ===")
top_vendors = df['vendorProject'].value_counts().head(10)
print(top_vendors)

# Date analysis
df['dateAdded'] = pd.to_datetime(df['dateAdded'])
df['dueDate'] = pd.to_datetime(df['dueDate'])
df['days_to_due'] = (df['dueDate'] - df['dateAdded']).dt.days

print(f"\nAverage days from addition to due date: {df['days_to_due'].mean():.1f} days")

# Data cleaning and feature engineering
print("=== DATA PRE-PROCESSING ===")

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values
print("Handling missing values...")
df_processed['cwes'] = df_processed['cwes'].fillna('Unknown')
df_processed['notes'] = df_processed['notes'].fillna('No notes')

# Feature engineering
print("Creating new features...")

# Extract year and month
df_processed['year_added'] = df_processed['dateAdded'].dt.year
df_processed['month_added'] = df_processed['dateAdded'].dt.month

# Create binary target for ransomware use
df_processed['ransomware_target'] = (df_processed['knownRansomwareCampaignUse'] == 'Known').astype(int)

# Create severity indicators based on vulnerability type
def extract_severity_indicator(vuln_name):
    vuln_name_lower = str(vuln_name).lower()
    if any(term in vuln_name_lower for term in ['remote code execution', 'arbitrary code execution', 'code injection']):
        return 'Critical'
    elif any(term in vuln_name_lower for term in ['privilege escalation', 'buffer overflow', 'sql injection']):
        return 'High'
    elif any(term in vuln_name_lower for term in ['denial of service', 'authentication bypass', 'information disclosure']):
        return 'Medium'
    else:
        return 'Low'

df_processed['severity_indicator'] = df_processed['vulnerabilityName'].apply(extract_severity_indicator)

# Create vendor risk profile (vendors with many vulnerabilities might be higher risk)
vendor_risk = df_processed['vendorProject'].value_counts()
df_processed['vendor_risk_score'] = df_processed['vendorProject'].map(vendor_risk)

# Text length features (proxy for complexity)
df_processed['short_desc_length'] = df_processed['shortDescription'].str.len()
df_processed['vuln_name_length'] = df_processed['vulnerabilityName'].str.len()

print("Pre-processing completed!")
print(f"Processed dataset shape: {df_processed.shape}")

import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# LightGBM (Optionnel mais recommandé)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM non détecté. Le modèle basculera sur RandomForest.")

warnings.filterwarnings("ignore")

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
# Les chemins sont relatifs pour fonctionner directement après MainMLProject.py
BASE_DIR = Path(".")

@dataclass
class Config:
    patched_path: str = str(BASE_DIR / "dataset_patched.csv")
    unpatched_path: str = str(BASE_DIR / "dataset_unpatched.csv")
    output_path: str = str(BASE_DIR / "predictions_risk_report.csv")
    
    # Noms de colonnes alignés avec MainMLProject.py
    target_col: str = "is_kev"
    text_col: str = "shortDescription"
    date_col: str = "dateAdded"
    cvss_score_col: str = "cvss_score"
    id_col: str = "cve_id"
    
    # Paramètres ML
    test_size_frac: float = 0.2
    random_state: int = 42

CFG = Config()

# ==============================================================================
# --- 2. FEATURE ENGINEERING ---
# ==============================================================================

KEYWORD_PATTERNS = {
    "has_rce": r"\b(remote code execution|rce|code execution)\b",
    "has_auth_bypass": r"\b(auth(entication)? bypass|bypass auth|authorization bypass)\b",
    "has_deserialization": r"\b(deserialization|unsafe deserialization)\b",
    "has_sqli": r"\b(sql injection|sqli)\b",
    "has_xss": r"\b(cross[-\s]?site scripting|xss)\b",
    "has_path_traversal": r"\b(path traversal|directory traversal)\b",
    "has_privesc": r"\b(privilege escalation|privesc|elevation of privilege)\b",
}

def add_description_flags(df: pd.DataFrame, text_col: str):
    """Crée des drapeaux binaires basés sur des mots-clés dans la description."""
    desc = df[text_col].fillna("").astype(str)
    for col, pat in KEYWORD_PATTERNS.items():
        df[col] = desc.str.contains(pat, flags=re.IGNORECASE, regex=True).astype(int)
    return df

def add_date_features(df: pd.DataFrame, date_col: str):
    """Extrait l'âge de la vulnérabilité et les métadonnées temporelles."""
    # Conversion forcée en UTC pour éviter les erreurs de timezone
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    
    # Si des dates sont NaT (Not a Time), on remplit par la médiane ou une date par défaut
    if df[date_col].isnull().any():
        median_date = df[date_col].median()
        df[date_col] = df[date_col].fillna(median_date)

    df["pub_year"] = df[date_col].dt.year
    df["pub_month"] = df[date_col].dt.month
    
    today = pd.Timestamp(datetime.now(timezone.utc))
    df["age_days"] = (today - df[date_col]).dt.days
    return df

def add_desc_len(df: pd.DataFrame, text_col: str):
    """Calcule la longueur de la description (proxy de complexité)."""
    df["desc_len"] = df[text_col].fillna("").astype(str).str.len()
    return df

def engineer_features(df: pd.DataFrame):
    """Pipeline complet de création de features."""
    df = add_description_flags(df, CFG.text_col)
    df = add_date_features(df, CFG.date_col)
    df = add_desc_len(df, CFG.text_col)
    return df

def temporal_train_test_split(df: pd.DataFrame, date_col: str, test_frac: float):
    """Split temporel strict (on apprend sur le passé, on teste sur le futur)."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n_test = int(len(df) * test_frac)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    return train_df, test_df

# ==============================================================================
# --- 3. MODÉLISATION ---
# ==============================================================================

def get_feature_columns(df: pd.DataFrame):
    """Sélectionne dynamiquement les colonnes numériques et catégorielles disponibles."""
    # Features numériques de base
    numeric_cols = [CFG.cvss_score_col, "desc_len", "age_days", "pub_year"]
    
    # Ajout des drapeaux booléens (traités comme numériques)
    flag_cols = list(KEYWORD_PATTERNS.keys())
    numeric_cols += [c for c in flag_cols if c in df.columns]

    # Features catégorielles (ex: Vendor si disponible et pertinent, ici on reste simple)
    categorical_cols = [] 
    if "vendorProject" in df.columns:
        # On ne garde vendor que s'il n'y en a pas trop, sinon risque d'overfitting
        # Pour ce script, on l'ignore pour généraliser sur la description/CVSS
        pass

    # Filtrage final pour ne garder que ce qui existe vraiment dans le DF
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    return numeric_cols, categorical_cols

def build_pipeline(df: pd.DataFrame, y_train: pd.Series):
    """Construit le pipeline Scikit-Learn (Preprocessing + Modèle)."""
    numeric_cols, categorical_cols = get_feature_columns(df)

    # 1. Définition du Modèle
    if HAS_LIGHTGBM:
        # Gestion du déséquilibre de classe
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        spw = neg / max(pos, 1)

        clf = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=CFG.random_state,
            scale_pos_weight=spw,
            n_jobs=-1,
            verbose=-1
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=CFG.random_state,
            n_jobs=-1
        )

    # 2. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            # NLP sur la description
            ("text", TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2)
            ), CFG.text_col),

            # Standardisation des numériques
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
        ],
        remainder="drop"
    )

    # 3. Pipeline Global
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    return pipe

# ==============================================================================
# --- 4. EXÉCUTION ---
# ==============================================================================

if __name__ == "__main__":
    print("\n=== Lancement du module ML (Lisa.py) ===")
    
    # 1. Chargement des données générées par MainMLProject.py
    try:
        patched = pd.read_csv(CFG.patched_path)
        unpatched = pd.read_csv(CFG.unpatched_path)
        
        # S'assurer que la target est correcte
        patched[CFG.target_col] = 1
        unpatched[CFG.target_col] = 0
        
        df = pd.concat([patched, unpatched], ignore_index=True)
        print(f"Données chargées : {len(df)} vulnérabilités.")
        print(f" - Patched (KEV) : {len(patched)}")
        print(f" - Unpatched     : {len(unpatched)}")
        
    except FileNotFoundError:
        print("ERREUR : Les fichiers 'dataset_patched.csv' ou 'dataset_unpatched.csv' sont introuvables.")
        print("Veuillez exécuter MainMLProject.py en premier pour les générer.")
        exit()

    # 2. Feature Engineering
    print("Génération des features (NLP, Temporelles, Regex)...")
    df = engineer_features(df)
    
    # Nettoyage des lignes sans date (critique pour le split temporel)
    df = df.dropna(subset=[CFG.date_col]).reset_index(drop=True)

    # 3. Split Train/Test
    train_df, test_df = temporal_train_test_split(df, CFG.date_col, CFG.test_size_frac)
    
    X_train = train_df
    y_train = train_df[CFG.target_col].astype(int)
    X_test = test_df
    y_test = test_df[CFG.target_col].astype(int)
    
    print(f"Train set : {len(train_df)} | Test set : {len(test_df)}")

    # 4. Entraînement
    print("Entraînement du modèle en cours...")
    pipeline = build_pipeline(train_df, y_train)
    pipeline.fit(X_train, y_train)

    # 5. Évaluation
    print("\n--- Évaluation sur le Test Set (Futur) ---")
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba_test)
    prec = precision_score(y_test, pred_test, zero_division=0)
    rec = recall_score(y_test, pred_test, zero_division=0)

    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred_test, digits=3))

    # 6. Prédiction de risque sur les non-KEV (Le but du projet)
    print("--- Calcul des scores de risque pour les CVE non exploitées ---")
    
    # On reprend toutes les données non exploitées (target=0)
    non_kev_df = df[df[CFG.target_col] == 0].copy()
    
    # Prédiction de probabilité
    non_kev_df["risk_score"] = pipeline.predict_proba(non_kev_df)[:, 1]
    
    # Définition des tiers de risque basés sur les quantiles
    q99 = non_kev_df["risk_score"].quantile(0.99)
    q95 = non_kev_df["risk_score"].quantile(0.95)
    q75 = non_kev_df["risk_score"].quantile(0.75)
    
    def get_tier(score):
        if score >= q99: return "CRITICAL (Top 1%)"
        if score >= q95: return "HIGH (Top 5%)"
        if score >= q75: return "MEDIUM (Top 25%)"
        return "LOW"

    non_kev_df["risk_tier"] = non_kev_df["risk_score"].apply(get_tier)
    
    # Sauvegarde des résultats
    output_cols = [CFG.id_col, "risk_score", "risk_tier", CFG.cvss_score_col, "pub_year"]
    # Ajout du titre si dispo, sinon description
    if "vulnerabilityName" in non_kev_df.columns:
        output_cols.insert(1, "vulnerabilityName")
        
    final_report = non_kev_df[output_cols].sort_values("risk_score", ascending=False)
    
    final_report.to_csv(CFG.output_path, index=False)
    
    print(f"\nAnalyse terminée. Rapport sauvegardé sous : {CFG.output_path}")
    print("Top 5 CVEs à risque élevé détectées (non encore KEV) :")
    print(final_report.head(5)[[CFG.id_col, "risk_score", "risk_tier"]])
