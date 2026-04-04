import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib
import os

# --- CONFIG ---
DATA_CSV = "hr_data.csv"
DATA_PARQUET = "hr_data.parquet"
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

# 1. DATA PREPARATION & OPTIMIZATION
if not os.path.exists(DATA_PARQUET):
    print(f"Parquet not found. Converting {DATA_CSV} to Parquet...")
    pd.read_csv(DATA_CSV).to_parquet(DATA_PARQUET)
    print("Conversion complete.")

print("Loading Parquet data...")
df = pd.read_parquet(DATA_PARQUET)

# 2. FEATURE ENGINEERING
print("Engineering features...")
df['Is_Attrition'] = df['Status'].apply(lambda x: 1 if x in ['Resigned', 'Terminated'] else 0)

le_dept = LabelEncoder()
le_mode = LabelEncoder()
df['Dept_Code'] = le_dept.fit_transform(df['Department'])
df['Mode_Code'] = le_mode.fit_transform(df['Work_Mode'])

joblib.dump(le_dept, 'le_dept.pkl')
joblib.dump(le_mode, 'le_mode.pkl')

FEATURES = ['Salary_INR', 'Performance_Rating', 'Experience_Years', 'Dept_Code', 'Mode_Code']
TARGET = 'Is_Attrition'

# 3. TRAIN / TEST SPLIT
print(f"Sampling {SAMPLE_SIZE:,} rows for training...")
train_df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
X = train_df[FEATURES]
y = train_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# 4. MODEL TRAINING
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y_train)

# 5. MODEL EVALUATION
print("\n--- Model Evaluation on Hold-out Test Set ---")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
auc_roc   = roc_auc_score(y_test, y_proba)

print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc_roc:.4f}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Attrited']))

# 6. PREDICT ON ALL 2M EMPLOYEES
print("Calculating Attrition Probability for all 2,000,000 employees...")
X_full = df[FEATURES]
df['Attrition_Probability'] = model.predict_proba(X_full)[:, 1]

# 7. FEATURE IMPORTANCE
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n--- Top Attrition Drivers (Feature Importance) ---")
print(importances)

# 8. SAVE ARTIFACTS
joblib.dump(model, 'attrition_model.pkl')
df.to_parquet('data_processed_ml.parquet')
print("\nPipeline complete.")
print("  Model saved    -> attrition_model.pkl")
print("  Enriched data  -> data_processed_ml.parquet")
