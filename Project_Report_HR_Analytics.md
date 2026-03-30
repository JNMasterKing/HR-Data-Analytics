# Comprehensive HR Data Analytics Report

## 1. Project Overview
This report provides a detailed analysis of a global HR dataset containing **2,000,000 employee records**. The objective of this work was to move from basic data exploration to strategic workforce planning by identifying attrition trends and building a risk-prediction framework.

---

## 2. Dataset Description
The dataset consists of 11 primary dimensions:
- **Demographics:** `Full_Name`, `Location`, `Country` (Engineered).
- **Organization:** `Department`, `Job_Title`, `Hire_Date`, `Year` (Engineered).
- **Performance & Tenure:** `Performance_Rating` (1-5), `Experience_Years` (0-15).
- **Employment Status:** `Status` (Active, Resigned, Retired, Terminated), `Work_Mode` (On-site, Remote).
- **Compensation:** `Salary_INR`.

---

## 3. Core Objectives
1. **Workforce Profiling:** Understand the structure and growth of the organization.
2. **Compensation Analysis:** Identify pay drivers and internal equity.
3. **Attrition Diagnostics:** Quantify employee loss and identify high-leakage areas.
4. **Predictive Risk Modeling:** Segment the active workforce into risk categories to enable proactive HR intervention.

---

## 4. Phase 1: Exploratory Data Analysis (EDA) Insights

### 4.1 Workforce Composition
- **IT Dominance:** The organization is tech-heavy, with the **IT Department** being the largest (over 600k employees). **Software Engineer** is the most frequent job title.
- **Modern Work Culture:** A significant **40%** of the workforce operates **Remote**, while **60%** are **On-site**.
- **Tenure Concentration:** Most employees have **0–4 years** of experience, indicating a relatively young or high-growth organization.

### 4.2 Hiring Trends
- Analysis of `Hire_Date` revealed a **massive growth surge starting in 2015**, peaking between 2021 and 2024. This suggests a period of rapid global expansion.

### 4.3 The Performance-Salary Paradox
- **Key Discovery:** There is **nearly zero correlation (-0.0002)** between `Performance_Rating` and `Salary_INR`. 
- **Finding:** Pay is driven by **Role** and **Experience** rather than merit-based performance scores. This represents a potential area for policy optimization.

---

## 5. Phase 2: Advanced Attrition Analysis

### 5.1 Attrition Rate Metrics
We calculated the **Attrition Rate** (Resigned + Terminated) across the organization:
- **Overall Attrition Rate:** **24.93%**
- **Top Attrition Departments:**
    1. **Finance:** 25.13%
    2. **Sales:** 24.98%
    3. **R&D:** 24.98%

### 5.2 Work Mode Impact
The analysis suggests that work mode (Remote vs. On-site) has a measurable impact on retention, which is visualized in the generated `attrition_analysis_report.png`.

---

## 6. Phase 3: Strategic Risk Profiling (Predictive Model)

Since standard ML libraries were not initially available, we developed a **Heuristic Predictive Model** based on **Relative Salary** (Salary vs. Role Median) and **Performance Ratings**.

### 6.1 Risk Segments Identified (Active Workforce)
We categorized the 1.4M active employees into three distinct profiles:

1. **High (Flight Risk) - 157,784 Employees (~11.2%)**:
   - **Profile:** High performers (Rating 4-5) who are paid below 80% of the median for their role.
   - **Risk:** Highly likely to be headhunted or resign due to compensation dissatisfaction.

2. **High (Performance Gap) - 158,191 Employees (~11.3%)**:
   - **Profile:** Low performers (Rating 1-2) who are paid above 120% of the role median.
   - **Risk:** High-cost employees with low output; candidates for PIP (Performance Improvement Plans) or restructuring.

3. **Low Risk - 1,085,583 Employees (~77.5%)**:
   - **Profile:** Employees whose performance aligns with their compensation.

---

## 7. Actionable Recommendations
1. **Retention Focus:** HR should prioritize the **157k "Flight Risk"** employees for immediate salary reviews or retention bonuses to prevent the loss of top talent.
2. **Performance Alignment:** Implement a merit-based pay structure to fix the "Zero-Correlation" issue between ratings and salary.
3. **Departmental deep-dive:** Investigate the **Finance** department to understand why it has the highest attrition rate (25.13%) compared to IT.

---

## 8. Technical Summary
- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn.
- **Outputs Produced:**
    - `Analysis_Insights.md` (Initial EDA)
    - `hr_attrition_analysis.py` (Analysis Script)
    - `attrition_analysis_report.png` (Visual Report)
    - `hr_dashboard.py` (Dashboard Script)
    - `Project_Report_HR_Analytics.md` (Final Documentation)
