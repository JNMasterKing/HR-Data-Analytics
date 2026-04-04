# 📋 Final Strategic Analysis Report

> **Project:** HR Data Analytics — Predicting Attrition for 2M Employees  
> **Author:** Jaynarayan Marwadi  
> **Date:** April 2026

---

## 1. Executive Summary

This report consolidates findings from a 4-stage data science pipeline applied to a 2-million-employee HR dataset. The core objective was to identify attrition drivers, build a predictive model, and recommend financially quantified retention strategies.

**Bottom Line:** A targeted 15% salary increase for high-performing, underpaid employees is projected to reduce attrition by a significant margin, saving the organization millions in rehiring costs.

---

## 2. Problem Statement

| Metric | Value |
|--------|-------|
| Total Employees | 2,000,000 |
| Overall Attrition Rate | 24.93% |
| High Flight Risk (Active) | 157,784 |
| Performance–Salary Correlation | ~0% |
| Top Attrition Department | Finance (25.13%) |

---

## 3. Root Cause Analysis

### 3.1 Pay-for-Performance Disconnect
The most alarming finding is a **near-zero correlation between Performance Rating and Salary**. Top-rated employees (Rating 4–5) are frequently paid below the median for their job title. This creates a classic "Flight Risk" scenario — talented employees feel undervalued and leave.

### 3.2 Feature Importance (Random Forest Model)
| Driver | Importance |
|--------|------------|
| Salary | 64% |
| Tenure (Experience Years) | 16% |
| Performance Rating | 8% |
| Department | 7% |
| Work Mode | 3% |
| Other | 2% |

Salary dominates at **64%** of the model's predictive power — confirming the pay policy is the single most actionable lever.

### 3.3 Departmental Hotspots
Finance leads attrition at **25.13%**, followed closely by other departments. The organization should prioritize retention programs in Finance, where critical financial knowledge is walking out the door.

---

## 4. Recommendations

### Recommendation 1: Salary Correction for High Performers
- **Action:** Increase salary by 15% for employees with Performance Rating ≥ 4 AND Salary < 80% of role median.
- **Estimated Impact:** Reduces Flight Risk pool by ~35%, saving millions in rehiring (industry avg: 1.5× salary per exit).
- **Priority:** CRITICAL — implement within Q2.

### Recommendation 2: Structured Performance-Pay Review
- **Action:** Establish a bi-annual pay review tied explicitly to performance ratings.
- **Goal:** Eliminate the 0% performance–salary correlation over 2 years.

### Recommendation 3: Finance Department Retention Program
- **Action:** Conduct targeted stay-interviews in Finance; introduce team-level retention bonuses.
- **Goal:** Reduce Finance attrition from 25.13% to below the company average.

### Recommendation 4: Early Tenure Engagement
- **Action:** Strengthen onboarding and 12-month check-ins, as attrition risk is highest in the first 4 years.

---

## 5. Financial Model

```
Average Salary (INR)         : ₹X (from dataset)
Cost per Attrition Exit      : 1.5 × Avg Salary
Current Annual Attrition     : ~498,600 employees (24.93% of 2M)
Projected Reduction (15% raise): ~35% fewer exits
Estimated Annual Saving      : ₹[calculated dynamically in dashboard]
```

Use the **What-If Simulator** in the Streamlit dashboard (`04_strategic_hr_dashboard.py`) to model live financial projections with adjustable salary increase percentages.

---

## 6. Conclusion

The data tells a clear story: **this organization pays for titles, not performance.** The fix is financially straightforward — targeted pay correction for high performers costs a fraction of the perpetual rehiring cycle. The ML model and heuristic risk engine built in this project give HR leadership the precision tools to act surgically rather than broadly.
