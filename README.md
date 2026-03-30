#  HR Data Analytics

A comprehensive data science project analyzing **2,000,000 employee records** to identify workforce trends, salary inequities, and attrition risks. This project moves beyond simple EDA to provide a **Strategic Risk-Profiling Model** for proactive talent management.

---

## 📊 Project Highlights
- **Scale:** Processes and visualizes a massive dataset of 2M records.
- **Key Discovery:** Identified a **"Performance-Salary Paradox"** where salary correlates with experience/role but has near-zero correlation with performance ratings.
- **Attrition Analytics:** Detailed breakdown of the **24.93% overall attrition rate**, pinpointing Finance and Sales as high-risk departments.
- **Strategic Modeling:** A heuristic-based **Risk Profiling Model** that segments employees into:
    - 🚨 **High Flight Risk:** Top performers who are underpaid.
    - ⚠️ **Performance Gap:** Low performers with high compensation.
    - ✅ **Low Risk:** Balanced workforce core.

---

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Interactive Dashboard:** Streamlit
- **Documentation:** Markdown (Detailed Technical Reports Included)

---

## 📁 Repository Structure
- `hr_dashboard.py`: Interactive Streamlit dashboard with Plotly visuals.
- `hr_attrition_analysis.py`: Standalone script for attrition rates and risk scoring.
- `Project_Report_HR_Analytics.md`: **Final Detailed Technical Report.**
- `Analysis_Insights.md`: Initial findings from the EDA phase.
- `attrition_analysis_report.png`: Static visual summary of the attrition study.
- `HR_Data_Analysis_with_Python.ipynb`: Initial exploration and data cleaning.

---

## 🚀 How to Run the Interactive Dashboard

1. **Clone the repository:**
   ```bash
   git clone <YOUR_REPO_URL>
   cd HR_Data_Analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide the dataset:**
   Place your `hr_data.csv` (2,000,000 rows) in the root directory.

4. **Launch the Dashboard:**
   ```bash
   streamlit run hr_dashboard.py
   ```

---

## 📈 Key Visuals
The dashboard provides four major views:
1. **Workforce KPI Metrics:** Real-time counters for Headcount, Attrition Rate, and Avg Salary.
2. **Departmental Attrition Heatmap:** Identifying turnover hotspots.
3. **Risk Profile Donut Chart:** Visualizing the health of the active workforce.
4. **Interactive Salary Scatter:** Exploring pay equity across experience levels and performance ratings.

---

## 📜 Findings Summary
- **Hyper-Growth:** The company saw a massive hiring surge starting in 2015.
- **Remote vs On-site:** 40% of the workforce is remote, impacting retention strategies.
- **Attrition Warning:** Finance is currently the most vulnerable department with a 25.13% attrition rate.

---
