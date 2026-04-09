# HR Data Analytics: 2M Employee Attrition

![Python](https://img.shields.io/badge/python-3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Parquet](https://img.shields.io/badge/Parquet-0072C6?style=for-the-badge&logo=apacheparquet&logoColor=white)

---

> **"Transforming HR from a cost center into a strategic profit-driver through Predictive ROI Modeling."**

This project analyzes a massive workforce of **2,000,000 employees** to solve one of the most expensive problems in business: **Talent Attrition.** By combining Big Data (Parquet) and Machine Learning (Random Forest), I built a "What-If" simulator that helps HR Directors save millions in turnover costs.

---

## 📌 Business Case: The High Cost of Turnover
*   **Massive Scale:** Analyzing 2 million records (Standard Excel tools crash at this volume).
*   **Turnover Crisis:** The organization currently faces a **24.93% attrition rate**.
*   **The Pay Paradox:** Top performers (Rating 4-5) often earn less than the median, creating a massive flight risk.
*   **Financial Impact:** Hiring and training a replacement costs **1.5x an employee's annual salary**.

## 💡 The Technical Solution
*   **Cloud-Optimized Data Architecture:** Used **Parquet** for 10x faster I/O and aggregate-sampling for the dashboard to ensure sub-second response times.
*   **Predictive Modeling:** Trained a **Random Forest Classifier** to identify key drivers.
*   **Financial Simulation:** A real-time **ROI Engine** that predicts how salary adjustments impact the bottom line.

---

## 📊 Model Performance & Insights
Our AI model doesn't just predict; it explains **why** people leave.

| Metric | Value |
| :--- | :--- |
| **Model Accuracy** | **75.14%** |
| **Primary Driver** | **Salary (64% Importance)** |
| **Secondary Driver** | **Tenure (16% Importance)** |
| **Tertiary Driver** | **Performance (8% Importance)** |

> **Key Discovery:** The model identified that **Performance and Salary have a ~0% correlation**, suggesting that the company's current pay structure does not reward top talent, which is the root cause of high-performer attrition.

---

## 🕵️‍♂️ Key EDA Discoveries
Before building the AI, a deep-dive Exploratory Data Analysis revealed:
*   **The Tenure Wall:** Most attrition happens in the first **4 years** of employment.
*   **The Remote Factor:** **40% of the workforce** is Remote. Surprisingly, Remote workers show a different attrition pattern than On-site staff.
*   **Departmental Volatility:** **Finance** and **Tech** have the highest resignation rates, while **HR** and **Legal** are more stable.
*   **Workload Impact:** High-performing employees are often given 2x the workload without a corresponding salary increase.

---

## 📂 Project Structure (Industry Standard)
```text
HR_Data_Analytics/
├── data/                  # Big Data: .parquet and .csv files
├── models/                # Artifacts: Trained .pkl models and LabelEncoders
├── notebooks/             # Research: 01_exploratory_analysis.ipynb
├── reports/               # Insights: Final analysis and visualization reports
├── 02_train_model.py      # ML Pipeline: Training & Parquet Optimization
├── 03_risk_analysis.py    # Heuristics: identifying pay gaps and flight risks
├── 04_hr_dashboard.py     # Frontend: The Strategic ROI Simulator (Streamlit)
├── 05_data_prep.py        # ETL: Preparing aggregate data for the dashboard
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Usage

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Training & Data Preparation:**
    ```bash
    python 02_train_attrition_model.py
    python 05_prepare_dashboard_data.py
    ```

3.  **Launch Simulator:**
    ```bash
    streamlit run 04_strategic_hr_dashboard.py
    ```

---
**Developed by [Jaynarayan] | HR Data Analytics Internship Project**
