import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Strategic HR Modeling & Simulation", layout="wide")

# --- DATA LOADING (Cloud-Optimized: NO 2M row parquet) ---
@st.cache_data
def load_data():
    dept_df   = pd.read_parquet("agg_department.parquet")
    mode_df   = pd.read_parquet("agg_workmode.parquet")
    risk_df   = pd.read_parquet("high_risk_sample.parquet")
    feat_df   = pd.read_parquet("agg_feature_importance.parquet")
    salary_df = pd.read_parquet("agg_salary_dist.parquet")
    return dept_df, mode_df, risk_df, feat_df, salary_df

@st.cache_resource
def load_model():
    model   = joblib.load("attrition_model.pkl")
    le_dept = joblib.load("le_dept.pkl")
    le_mode = joblib.load("le_mode.pkl")
    return model, le_dept, le_mode

st.title("🚀 Strategic HR Modeling & 'What-If' Simulation")
st.markdown("Optimize your workforce using **Machine Learning (Random Forest)** and **Real-Time Retention Simulation**.")

dept_df, mode_df, risk_df, feat_df, salary_df = load_data()
model, le_dept, le_mode = load_model()

# --- SIDEBAR: SIMULATION & FILTERS ---
st.sidebar.header("🕹️ 'What-If' Retention Simulator")
st.sidebar.info("Simulate how a salary increase for High Performers (Rating 4-5) affects retention.")

salary_increase_pct = st.sidebar.slider("Salary Increase % for High Performers", 0, 50, 0)
attrition_threshold = st.sidebar.slider("Attrition Probability Threshold", 0.1, 0.9, 0.3)

st.sidebar.divider()
st.sidebar.header("Global Filters")
dept_filter = st.sidebar.multiselect(
    "Filter by Department",
    options=dept_df["Department"].unique(),
    default=dept_df["Department"].unique()
)

# --- SIMULATION LOGIC (uses 10k Active high-risk sample + real ML model) ---
sim_df = risk_df[risk_df["Department"].isin(dept_filter)].copy()

# Snapshot original salary BEFORE raise (Fix #4)
avg_salary_original = sim_df["Salary_INR"].mean()

# Apply raise only to High Performers
sim_df.loc[sim_df["Performance_Rating"] >= 4, "Salary_INR"] *= (1 + salary_increase_pct / 100)

# Re-run real ML model on simulated salaries (Fix #3)
sim_df["Dept_Code"] = le_dept.transform(sim_df["Department"])
sim_df["Mode_Code"] = le_mode.transform(sim_df["Work_Mode"])
X_sim = sim_df[["Salary_INR", "Performance_Rating", "Experience_Years", "Dept_Code", "Mode_Code"]]
sim_df["New_Attrition_Prob"] = model.predict_proba(X_sim)[:, 1]

# Current vs Simulated counts
current_at_risk   = len(sim_df[sim_df["Attrition_Probability"] > attrition_threshold])
simulated_at_risk = len(sim_df[sim_df["New_Attrition_Prob"]   > attrition_threshold])
retention_gain    = current_at_risk - simulated_at_risk

# Cost calculation uses original (pre-raise) salary baseline (Fix #4)
cost_per_exit = avg_salary_original * 1.5
total_savings = retention_gain * cost_per_exit

# Retention gain % with correct sign (Fix #8)
gain_pct = (retention_gain / max(1, current_at_risk)) * 100

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Flight Risks",   f"{current_at_risk:,}")
col2.metric("Simulated Flight Risks", f"{simulated_at_risk:,}",
            delta=f"{-retention_gain:,}", delta_color="inverse")
col3.metric("Retention Rate Gain",    f"{gain_pct:+.1f}%")
col4.metric("💰 Est. Cost Savings",   f"₹{total_savings:,.0f}")

st.divider()

# --- INTERACTIVE VISUALS ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Model Insights: Key Drivers of Attrition")
    # Fix #5: real feature importances from model, NOT hardcoded
    fig_imp = px.bar(
        feat_df, x="Feature", y="Importance",
        labels={"Importance": "Importance Score"},
        color_discrete_sequence=["#636EFA"],
        template="plotly_white"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with row1_col2:
    st.subheader("Simulated Attrition Reduction (Probability Distribution)")
    # Fix #1/#2: uses only 10k-row sim_df — no 2M rows in memory
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=sim_df["Attrition_Probability"], name="Current Risk",
        opacity=0.5, nbinsx=50, marker_color="#EF553B"
    ))
    fig_dist.add_trace(go.Histogram(
        x=sim_df["New_Attrition_Prob"], name="Simulated Risk",
        opacity=0.5, nbinsx=50, marker_color="#00CC96"
    ))
    fig_dist.update_layout(barmode="overlay", template="plotly_white")
    st.plotly_chart(fig_dist, use_container_width=True)

# --- DEPARTMENT BREAKDOWN: Voluntary vs Involuntary (Fix #6) ---
st.subheader("Attrition by Department: Voluntary vs Involuntary")
dept_filtered = dept_df[dept_df["Department"].isin(dept_filter)]
fig_dept = px.bar(
    dept_filtered, x="Department",
    y=["resignations", "terminations"],
    barmode="group",
    labels={"value": "Employee Count", "variable": "Type"},
    color_discrete_map={"resignations": "#EF553B", "terminations": "#636EFA"},
    template="plotly_white"
)
st.plotly_chart(fig_dept, use_container_width=True)

# --- HIGH-RISK ACTION LIST (Fix #9: active employees only) ---
st.subheader("Top High-Risk Active Employees (Action List)")
action_list = (
    sim_df
    .sort_values("New_Attrition_Prob", ascending=False)
    .head(1000)
)
st.dataframe(
    action_list[[
        "Department", "Job_Title", "Salary_INR",
        "Performance_Rating", "Experience_Years",
        "Attrition_Probability", "New_Attrition_Prob"
    ]],
    use_container_width=True
)

# --- EXPORT (Fix #12: real download button) ---
st.subheader("Export Simulation Report")
csv_bytes = action_list.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Report as CSV",
    data=csv_bytes,
    file_name="hr_simulation_report.csv",
    mime="text/csv"
)
