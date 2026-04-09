import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import os

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1. DATA PREPARATION & OPTIMIZATION
if not os.path.exists("data/hr_data.parquet"):
    print("Parquet file not found. Converting hr_data.csv to high-performance Parquet format...")
    if os.path.exists("data/hr_data.csv"):
        pd.read_csv("data/hr_data.csv").to_parquet("data/hr_data.parquet")
    elif os.path.exists("hr_data.csv"):
        pd.read_csv("hr_data.csv").to_parquet("data/hr_data.parquet")
    else:
        print("Error: hr_data.csv not found in root or data/ folder.")
    print("Conversion complete.")

print("Loading Parquet data...")
df = pd.read_parquet("data/hr_data.parquet")

# 2. FEATURE ENGINEERING
print("Engineering features...")
# Target: Resigned or Terminated
df['Is_Attrition'] = df['Status'].apply(lambda x: 1 if x in ['Resigned', 'Terminated'] else 0)

# Encode categorical variables for the ML model
le_dept = LabelEncoder()
le_mode = LabelEncoder()
df['Dept_Code'] = le_dept.fit_transform(df['Department'])
df['Mode_Code'] = le_mode.fit_transform(df['Work_Mode'])

# Save encoders to ensure consistency in the dashboard
joblib.dump(le_dept, 'models/le_dept.pkl')
joblib.dump(le_mode, 'models/le_mode.pkl')

# 3. SELECT FEATURES & TRAIN
# Using a 100k sample for balanced speed and accuracy during training
print("Training Random Forest Classifier (100k samples)...")
train_df = df.sample(100000, random_state=42)
X = train_df[['Salary_INR', 'Performance_Rating', 'Experience_Years', 'Dept_Code', 'Mode_Code']]
y = train_df['Is_Attrition']

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# 4. PREDICT ATTRITION PROBABILITY (on all 2 million)
print("Calculating Attrition Probability for all 2,000,000 employees...")
X_full = df[['Salary_INR', 'Performance_Rating', 'Experience_Years', 'Dept_Code', 'Mode_Code']]
df['Attrition_Probability'] = model.predict_proba(X_full)[:, 1]

# Save the model and the final ML-enriched dataset
joblib.dump(model, 'models/attrition_model.pkl')
df.to_parquet('data/data_processed_ml.parquet')

# 5. FEATURE IMPORTANCE
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Model Insights (Top Attrition Drivers) ---")
print(importances)

# 6. MODEL EVALUATION (Hold-out Test Set)
print("\nEvaluating model on hold-out test set...")

# Split training sample 80/20 for honest evaluation
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
eval_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
eval_model.fit(X_tr, y_tr)

y_pred      = eval_model.predict(X_test)
y_pred_prob = eval_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)
cm       = confusion_matrix(y_test, y_pred)
cr       = classification_report(y_test, y_pred, target_names=["Retained", "Attrition"])
tn, fp, fn, tp = cm.ravel()

report_lines = [
    "=" * 55,
    "  HR ATTRITION MODEL — EVALUATION REPORT",
    "=" * 55,
    "",
    "  Training Sample : 80,000 employees (from 100k sample)",
    "  Test Sample     : 20,000 employees (held-out, stratified)",
    "  Features        : Salary, Performance, Tenure, Dept, Work Mode",
    "",
    "-" * 55,
    "  OVERALL METRICS",
    "-" * 55,
    f"  Accuracy  : {accuracy * 100:.2f}%",
    f"  ROC-AUC   : {roc_auc:.4f}",
    "",
    "-" * 55,
    "  CLASSIFICATION REPORT",
    "-" * 55,
    cr,
    "-" * 55,
    "  CONFUSION MATRIX",
    "  Rows = Actual | Cols = Predicted",
    "-" * 55,
    f"                 Pred: Retained   Pred: Attrition",
    f"  Actual: Retained     {tn:<10}   {fp}",
    f"  Actual: Attrition    {fn:<10}   {tp}",
    "",
    f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}",
    "=" * 55,
]

report_text = "\n".join(report_lines)
with open("reports/model_metrics.txt", "w") as f:
    f.write(report_text)

print(report_text)

print("\nPipeline complete.")
print("- Model saved as           : models/attrition_model.pkl")
print("- Enriched Data saved as   : data/data_processed_ml.parquet")
print("- Evaluation Report saved  : reports/model_metrics.txt")
