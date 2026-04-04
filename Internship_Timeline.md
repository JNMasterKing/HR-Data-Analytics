# 🗓️ Internship Timeline — HR Data Analytics Project

> **Role:** Data Analytics Intern  
> **Author:** Jaynarayan Marwadi  
> **Duration:** March 30 – April 2026

---

## Week 1: Data Exploration & Setup

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 1 | Project briefing, dataset handover, environment setup | `requirements.txt`, repo initialized |
| Day 2 | Exploratory Data Analysis (EDA) — data profiling, null checks, distribution analysis | `01_exploratory_analysis.ipynb` |
| Day 3 | Deep dive: Performance vs Salary correlation, workforce demographics | `REPORT_EDA_Insights.md` |
| Day 4 | Big Data optimization — CSV → Parquet conversion, benchmarking load times | `hr_data.parquet` |
| Day 5 | Feature engineering, label encoding, ML pipeline design | Draft of `02_train_attrition_model.py` |

---

## Week 2: Machine Learning & Risk Analysis

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 6 | Random Forest model training (100k sample), initial accuracy evaluation | `attrition_model.pkl` |
| Day 7 | Full 2M inference, attrition probability scoring | `data_processed_ml.parquet` |
| Day 8 | Heuristic risk scoring — Flight Risk & Performance Gap identification | `03_heuristic_risk_analysis.py` |
| Day 9 | Visualization — 4-panel attrition analysis report | `attrition_analysis_report.png` |
| Day 10 | Review session with mentor — feedback on model accuracy and business framing | Model evaluation metrics added |

---

## Week 3: Dashboard & Strategic Reporting

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 11 | Streamlit dashboard architecture design, KPI layout planning | Dashboard wireframe |
| Day 12 | Implement What-If Salary Simulator and attrition probability distribution | `04_strategic_hr_dashboard.py` (v1) |
| Day 13 | Financial ROI modeling — cost-of-attrition vs cost-of-raise calculator | Dashboard KPIs finalized |
| Day 14 | Testing, deployment prep, README documentation | `README.md`, repo polished |
| Day 15 | Final presentation to HR Director — strategic recommendations delivered | `REPORT_Final_Strategic_Analysis.md` |

---

## Key Learnings

- **Big Data Handling:** Parquet format reduced data load time by ~10× vs CSV for 2M rows.
- **ML in Production:** Learned to separate training (sampled) from inference (full data) for speed without sacrificing coverage.
- **Business Translation:** Translating a model's feature importance into a financial recommendation ("64% of attrition is salary-driven") is as important as building the model.
- **Iterative Development:** Used Git branching and PRs to manage code changes cleanly across a multi-file pipeline.
