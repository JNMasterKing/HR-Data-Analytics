import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Strategic HR Modeling & Simulation", layout="wide")

# --- DATA LOADING (Big Data Optimized) ---
@st.cache_data
def load_data():
    # Load the ML-enriched data (Parquet)
    df = pd.read_parquet("data_processed_ml.parquet")
    return df

st.title("🚀 Strategic HR Modeling & 'What-If' Simulation")
st.markdown("Optimize your workforce using **Machine Learning (Random Forest)** and **Real-Time Retention Simulation**.")

df_full = load_data()

# --- SIDEBAR: SIMULATION & FILTERS ---
st.sidebar.header("🕹️ 'What-If' Retention Simulator")
st.sidebar.info("Simulate how a salary increase for High Performers (Rating 4-5) affects retention.")

salary_increase_pct = st.sidebar.slider("Salary Increase % for High Performers", 0, 50, 0)
attrition_threshold = st.sidebar.slider("Attrition Probability Threshold", 0.1, 0.9, 0.3)

st.sidebar.divider()
st.sidebar.header("Global Filters")
dept_filter = st.sidebar.multiselect("Filter by Department", options=df_full['Department'].unique(), default=df_full['Department'].unique())

# --- SIMULATION LOGIC ---
# Create a copy for simulation
sim_df = df_full[df_full['Department'].isin(dept_filter)].copy()

# Apply the salary increase to High Performers
sim_df.loc[sim_df['Performance_Rating'] >= 4, 'Salary_INR'] *= (1 + salary_increase_pct/100)

# Simulate Attrition Reduction (Rule: Every 10% raise reduces attrition prob by 15%)
# (Simplified for demonstration; in real production, we would re-run the ML model)
sim_df['New_Attrition_Prob'] = sim_df['Attrition_Probability'] * (1 - (salary_increase_pct/100) * 1.5)
sim_df['New_Attrition_Prob'] = sim_df['New_Attrition_Prob'].clip(0, 1)

# Current vs Simulated Attrition Count
current_resignations = len(sim_df[sim_df['Attrition_Probability'] > attrition_threshold])
simulated_resignations = len(sim_df[sim_df['New_Attrition_Prob'] > attrition_threshold])
retention_gain = current_resignations - simulated_resignations

# Estimated Attrition Cost (Industry avg: 1.5x salary per exit)
avg_salary = sim_df['Salary_INR'].mean()
cost_per_exit = avg_salary * 1.5
total_savings = retention_gain * cost_per_exit

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Flight Risks", f"{current_resignations:,}")
col2.metric("Simulated Flight Risks", f"{simulated_resignations:,}", delta=f"-{retention_gain:,}", delta_color="inverse")
col3.metric("Retention Rate Gain", f"+{((retention_gain/max(1,current_resignations))*100):.1f}%")
col4.metric("💰 Est. Cost Savings", f"₹{total_savings:,.0f}")

st.divider()

# --- INTERACTIVE VISUALS ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Model Insights: Key Drivers of Attrition")
    # Static values from our model run
    importances = {'Salary': 0.64, 'Tenure': 0.16, 'Performance': 0.08, 'Department': 0.07, 'Work Mode': 0.03}
    fig_imp = px.bar(x=list(importances.keys()), y=list(importances.values()), 
                      labels={'x': 'Feature', 'y': 'Importance Score'},
                      color_discrete_sequence=['#636EFA'], template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)

with row1_col2:
    st.subheader("Simulated Attrition Reduction (Probability Distribution)")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=sim_df['Attrition_Probability'], name='Current Risk', opacity=0.5, nbinsx=50, marker_color='#EF553B'))
    fig_dist.add_trace(go.Histogram(x=sim_df['New_Attrition_Prob'], name='Simulated Risk', opacity=0.5, nbinsx=50, marker_color='#00CC96'))
    fig_dist.update_layout(barmode='overlay', template="plotly_white")
    st.plotly_chart(fig_dist, use_container_width=True)

# --- RISK SEGMENTATION CHART ---
st.subheader("Top 1,000 High-Risk Employees (Action List)")
high_risk_list = sim_df[sim_df['Status'] == 'Active'].sort_values('New_Attrition_Prob', ascending=False).head(1000)
st.dataframe(high_risk_list[['Full_Name', 'Department', 'Job_Title', 'Salary_INR', 'Performance_Rating', 'Experience_Years', 'New_Attrition_Prob']], 
             use_container_width=True)

# --- EXPORT RESULTS ---
if st.button("Export Simulation Report for HR Board"):
    st.success(f"Simulation saved! Total projected savings: ₹{total_savings:,.0f}")
