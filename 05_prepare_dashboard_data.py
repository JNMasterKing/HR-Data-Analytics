import pandas as pd
import joblib
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Loading ML-enriched parquet (run locally only)...")
if os.path.exists("data/data_processed_ml.parquet"):
    df = pd.read_parquet("data/data_processed_ml.parquet")
else:
    print("Error: data/data_processed_ml.parquet not found. Please run 02_train_attrition_model.py first.")
    exit()

# 1. Department summary with voluntary vs involuntary split (Fix #6)
dept_summary = df.groupby("Department").agg(
    total        =("Is_Attrition", "count"),
    attrition    =("Is_Attrition", "sum"),
    resignations =("Status",    lambda x: (x == "Resigned").sum()),
    terminations =("Status",    lambda x: (x == "Terminated").sum()),
    avg_salary   =("Salary_INR",         "mean"),
    avg_perf     =("Performance_Rating",  "mean"),
    avg_exp      =("Experience_Years",    "mean"),
).reset_index()
dept_summary["attrition_rate"] = dept_summary["attrition"] / dept_summary["total"] * 100
dept_summary.to_parquet("data/agg_department.parquet", index=False)
print(f"  data/agg_department.parquet         -> {len(dept_summary)} rows")

# 2. Work-mode summary
mode_summary = df.groupby("Work_Mode").agg(
    total     =("Is_Attrition", "count"),
    attrition =("Is_Attrition", "sum"),
).reset_index()
mode_summary.to_parquet("data/agg_workmode.parquet", index=False)
print(f"  data/agg_workmode.parquet           -> {len(mode_summary)} rows")

# 3. High-risk ACTIVE employees only (Fix #9)
active_df = df[df["Status"] == "Active"]
high_risk = (
    active_df
    .nlargest(10000, "Attrition_Probability")
    [[
        "Department", "Job_Title", "Work_Mode",
        "Salary_INR", "Performance_Rating", "Experience_Years",
        "Attrition_Probability"
    ]]
)
high_risk.to_parquet("data/high_risk_sample.parquet", index=False)
print(f"  data/high_risk_sample.parquet       -> {len(high_risk)} rows")

# 4. Salary distribution (pre-binned — no raw float arrays to browser)
salary_dist = pd.cut(df["Salary_INR"], bins=30).value_counts().sort_index().reset_index()
salary_dist.columns = ["salary_bin", "count"]
salary_dist["salary_bin"] = salary_dist["salary_bin"].astype(str)
salary_dist.to_parquet("data/agg_salary_dist.parquet", index=False)
print(f"  data/agg_salary_dist.parquet        -> {len(salary_dist)} rows")

# 5. Real feature importances from trained model (Fix #5)
if os.path.exists("models/attrition_model.pkl"):
    model = joblib.load("models/attrition_model.pkl")
else:
    print("Error: models/attrition_model.pkl not found.")
    exit()

features = ["Salary_INR", "Relative_Salary", "Performance_Rating", "Experience_Years", "Dept_Code", "Mode_Code"]
feat_df  = (
    pd.Series(model.feature_importances_, index=features)
      .rename_axis("Feature")
      .reset_index(name="Importance")
      .sort_values("Importance", ascending=False)
)
feat_df["Feature"] = feat_df["Feature"].replace({
    "Salary_INR":         "Salary",
    "Relative_Salary":    "Relative Salary",
    "Performance_Rating": "Performance",
    "Experience_Years":   "Tenure",
    "Dept_Code":          "Department",
    "Mode_Code":          "Work Mode",
})
feat_df.to_parquet("data/agg_feature_importance.parquet", index=False)
print(f"  data/agg_feature_importance.parquet -> {len(feat_df)} rows")
