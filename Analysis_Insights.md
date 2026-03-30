# HR Data Analysis Insights

This document summarizes the key findings from the analysis of `HR_Data_Analysis_with_Python.ipynb` based on a dataset of 2,000,000 employee records.

## 1. Dataset Overview
- **Total Records:** 2,000,000
- **Columns:** Employee_ID, Full_Name, Department, Job_Title, Hire_Date, Location, Performance_Rating, Experience_Years, Status, Work_Mode, Salary_INR.

## 2. Key Visual Insights

### Workforce Composition
- **Experience:** High concentration in the 0–4 years bracket. Significant drop at 15 years.
- **Department Size:** **IT is the largest department**, followed by Sales and Operations. R&D is the smallest.
- **Work Mode:** **60% On-site** vs. **40% Remote**.
- **Job Titles:** **Software Engineer** is the most frequent role.

### Employee Lifecycle & Attrition
- **Current Status:** 70.1% Active, ~25% Attrition (Resigned + Terminated).
- **Hiring Trends:** Massive growth surge starting in 2015, peaking between 2021 and 2024.

### Compensation & Performance
- **Departmental Pay:** IT and Finance are the highest-paying departments; HR and Marketing are the lowest.
- **Role-based Pay:** Management and technical leads (IT/Finance Managers) command significantly higher salaries.
- **Performance Correlation:** There is **nearly zero correlation (-0.0002)** between Performance Rating and Salary, suggesting salary is driven by role/experience rather than annual performance scores.

## 3. Summary of Findings
- The organization is a large, tech-centric global entity.
- There is a clear "pay-by-role" structure rather than "pay-by-performance."
- The company underwent a period of hyper-growth in the last decade.
- Attrition is a significant factor, with 1 in 4 employees having left the company.
