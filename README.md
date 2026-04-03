# 🚀 HR Data Analytics: Predicting Attrition for 2M Employees

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Parquet](https://img.shields.io/badge/Parquet-0072C6?style=for-the-badge&logo=apacheparquet&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23FAAAE0.svg?style=for-the-badge&logo=jupyter&logoColor=white)

---

> **"Transforming static HR data into a high-performance financial decision tool."**

This project analyzes a massive workforce of **2,000,000 employees** to solve one of the most expensive problems in business: **Talent Attrition.** By combining Big Data (Parquet) and Machine Learning (Random Forest), I built a "What-If" simulator that helps HR Directors save millions in turnover costs.

---

## 📌 The Problem
*   **High Turnover:** The organization suffers from a **24.93% attrition rate.**
*   **Pay Inequity:** High performers are often underpaid, leading to "Flight Risk."
*   **Data Scale:** Standard tools (Excel) crash when handling 2 million records.

## 💡 The Solution
*   **Big Data Optimization:** Converted 2M rows to **Parquet** for 10x faster loading.
*   **Predictive AI:** Trained a **Random Forest model** to identify the top reasons for resignations.
*   **Strategic Dashboard:** Created an interactive **"What-If" Simulator** to predict how salary changes impact retention.

---

## 🛠️ Key Features
*   🎯 **Risk Profiling:** Identifies 157,784 "High Flight Risk" employees (Top talent likely to leave).
*   📊 **Interactive Dashboards:** Dynamic filtering by Department, Work Mode, and Performance.
*   💰 **Financial Forecasting:** Calculates the **ROI of retention** (cost of raises vs. cost of turnover).
*   🔍 **Feature Importance:** Explains *why* people leave (Salary is 64% of the driver).

---

## 📂 Project Pipeline
The project follows a professional 4-step data science workflow:
1.  **`01_exploratory_analysis.ipynb`** | Data cleaning & initial insights.
2.  **`02_train_attrition_model.py`** | ML Training & Big Data optimization (Parquet).
3.  **`03_heuristic_risk_analysis.py`** | Identifying pay gaps and statistical outliers.
4.  **`04_strategic_hr_dashboard.py`** | The final interactive Business Intelligence tool.

---

## 🚀 Quick Start

1.  **Clone & Install:**
    ```bash
    git clone <YOUR_REPO_URL>
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline (Training):**
    ```bash
    python 02_train_attrition_model.py
    ```

3.  **Launch the Dashboard:**
    ```bash
    streamlit run 04_strategic_hr_dashboard.py
    ```

---

## 📈 Strategic Insights
*   **The Pay Paradox:** Performance and Salary have a **0% correlation**, suggesting a disconnect in pay policy.
*   **Top Risk Dept:** **Finance** shows the highest attrition rate (25.13%).
*   **Cost Savings:** Increasing pay for high performers by 15% is projected to save the company **millions in re-hiring costs.**

---

## 📜 Full Documentation
*   [Final Strategic Report](./REPORT_Final_Strategic_Analysis.md)
*   [EDA Deep Dive](./REPORT_EDA_Insights.md)
*   [Internship Timeline](./Internship_Timeline.md)

---
**Created by [Jaynarayan] during my HR Data Analytics Internship.**
