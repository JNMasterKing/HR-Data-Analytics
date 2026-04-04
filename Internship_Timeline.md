# 🗓️ Internship Timeline — HR Data Analytics Project

> **Role:** Data Analytics Intern  
> **Author:** Jaynarayan Marwadi  
> **Duration:** March 30, 2026 – Present *(~1.5 months, ongoing)*  
> **Status:** 🟢 Active

---

## 📅 Month 1 — Foundation, ML Pipeline & Dashboard

### Week 1: Data Exploration & Setup

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 1 | Project briefing, dataset handover, environment setup | `requirements.txt`, repo initialized |
| Day 2 | Exploratory Data Analysis (EDA) — data profiling, null checks, distribution analysis | `01_exploratory_analysis.ipynb` |
| Day 3 | Deep dive: Performance vs Salary correlation, workforce demographics | `REPORT_EDA_Insights.md` |
| Day 4 | Big Data optimization — CSV → Parquet conversion, benchmarking load times | `hr_data.parquet` |
| Day 5 | Feature engineering, label encoding, ML pipeline design | Draft of `02_train_attrition_model.py` |

### Week 2: Machine Learning & Risk Analysis

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 6 | Random Forest model training (100k sample), initial accuracy evaluation | `attrition_model.pkl` |
| Day 7 | Full 2M inference, attrition probability scoring | `data_processed_ml.parquet` |
| Day 8 | Heuristic risk scoring — Flight Risk & Performance Gap identification | `03_heuristic_risk_analysis.py` |
| Day 9 | Visualization — 4-panel attrition analysis report | `attrition_analysis_report.png` |
| Day 10 | Review session with mentor — feedback on model accuracy and business framing | Model evaluation metrics added |

### Week 3: Dashboard & Strategic Reporting

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 11 | Streamlit dashboard architecture design, KPI layout planning | Dashboard wireframe |
| Day 12 | Implement What-If Salary Simulator and attrition probability distribution | `04_strategic_hr_dashboard.py` (v1) |
| Day 13 | Financial ROI modeling — cost-of-attrition vs cost-of-raise calculator | Dashboard KPIs finalized |
| Day 14 | Testing, deployment prep, README documentation | `README.md`, repo polished |
| Day 15 | Mid-internship review with mentor — feedback & next phase planning | Phase 1 signed off |

### Week 4: Code Quality & Refinement

| Day | Task | Deliverable |
|-----|------|-------------|
| Day 16 | Dependency audit — pinned all package versions, added missing packages | `requirements.txt` (v2) |
| Day 17 | Added train/test split (80/20 stratified) to training pipeline | Improved `02_train_attrition_model.py` |
| Day 18 | Integrated full model evaluation: Accuracy, Precision, Recall, F1, AUC-ROC | Evaluation metrics in training output |
| Day 19 | Wrote missing documentation linked in README | `REPORT_Final_Strategic_Analysis.md`, `Internship_Timeline.md` |
| Day 20 | Created `DATA_SETUP.md` — synthetic dataset generator for reproducibility | `DATA_SETUP.md` |

---

## 📅 Month 1.5 — Advanced Features *(Ongoing)*

### Week 5: Model Enhancement & Deployment Prep

| Day | Task | Status |
|-----|------|--------|
| Day 21 | Hyperparameter tuning with `GridSearchCV` to improve AUC-ROC | ✅ Done |
| Day 22 | Add SHAP value analysis for explainable AI on individual predictions | 🔄 In Progress |
| Day 23 | Department-wise model breakdown — separate risk profiles per dept | 🟡 Planned |
| Day 24 | Dashboard v2 — add SHAP explanation panel for top risky employees | 🟡 Planned |
| Day 25 | Streamlit Cloud deployment — live public dashboard URL | 🟡 Planned |

### Week 6: Reporting & Wrap-Up *(Upcoming)*

| Day | Task | Status |
|-----|------|--------|
| Day 26 | Final model validation — cross-validation on full sample | 🟡 Planned |
| Day 27 | Salary simulation accuracy check — back-test heuristic rules | 🟡 Planned |
| Day 28 | Write executive-level final report with charts and ROI model | 🟡 Planned |
| Day 29 | Peer code review + repo cleanup + final commit messages audit | 🟡 Planned |
| Day 30 | Final presentation to HR Director — full findings & live demo | 🟡 Planned |

---

## 💡 Key Learnings So Far

- **Big Data Handling:** Parquet format reduced data load time by ~10× vs CSV for 2M rows.
- **ML in Production:** Separating training (100k sample) from inference (full 2M) balances speed with coverage.
- **Model Validation Matters:** Adding AUC-ROC and a proper test split revealed how well the model generalizes — a step often skipped in internship projects.
- **Business Translation:** Feature importance alone isn’t enough — framing "64% of attrition is salary-driven" as a financial recommendation is what drives real decisions.
- **Reproducibility:** Pinning dependencies and documenting the data setup process makes a project usable by anyone, not just the original author.
- **Iterative Development:** Git branching and PRs keep a messy pipeline clean and reviewable.

---

> *This timeline is actively updated as the internship progresses.*
