import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="Strategic HR Insights Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_parquet("hr_data.parquet")    
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])
    # Calculate Risk Score Heuristics
    median_salary_role = df.groupby('Job_Title')['Salary_INR'].transform('median')
    df['Relative_Salary'] = df['Salary_INR'] / median_salary_role
    df['Risk_Score'] = 'Low'
    df.loc[(df['Performance_Rating'] >= 4) & (df['Relative_Salary'] < 0.8), 'Risk_Score'] = 'High (Flight Risk)'
    df.loc[(df['Performance_Rating'] <= 2) & (df['Relative_Salary'] > 1.2), 'Risk_Score'] = 'High (Performance Gap)'
    df['Is_Attrition'] = df['Status'].apply(lambda x: 1 if x in ['Resigned', 'Terminated'] else 0)
    return df

st.title("Strategic HR Insights & Attrition Dashboard")
st.markdown("Analyze workforce trends, salary equity, and attrition risk across 2,000,000 records.")

# --- SIDEBAR FILTERS ---
df_raw = load_data()
st.sidebar.header("Navigation & Filters")

dept_filter = st.sidebar.multiselect("Filter by Department", options=df_raw['Department'].unique(), default=df_raw['Department'].unique())
mode_filter = st.sidebar.multiselect("Work Mode", options=df_raw['Work_Mode'].unique(), default=df_raw['Work_Mode'].unique())
risk_filter = st.sidebar.multiselect("Risk Profile", options=df_raw['Risk_Score'].unique(), default=df_raw['Risk_Score'].unique())

# Apply filters
df = df_raw[
    (df_raw['Department'].isin(dept_filter)) & 
    (df_raw['Work_Mode'].isin(mode_filter)) &
    (df_raw['Risk_Score'].isin(risk_filter))
]

# --- KEY PERFORMANCE INDICATORS (KPIs) ---
col1, col2, col3, col4 = st.columns(4)

total_count = len(df)
overall_attrition_rate = (df['Is_Attrition'].mean() * 100)
avg_salary = df['Salary_INR'].mean()
high_risk_count = len(df[df['Risk_Score'] == 'High (Flight Risk)'])

col1.metric("Total Workforce", f"{total_count:,}")
col2.metric("Attrition Rate", f"{overall_attrition_rate:.2f}%", delta="-1.5%" if overall_attrition_rate < 25 else "High", delta_color="inverse")
col3.metric("Avg Salary (INR)", f"₹{avg_salary:,.0f}")
col4.metric(" High Flight Risk", f"{high_risk_count:,}")

st.divider()

# --- INTERACTIVE CHARTS ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Attrition Rate by Department (%)")
    dept_attr = df_raw.groupby('Department')['Is_Attrition'].mean().sort_values(ascending=False) * 100
    fig_dept = px.bar(dept_attr, x=dept_attr.index, y=dept_attr.values, 
                      labels={'y': 'Attrition %'}, color=dept_attr.values, 
                      color_continuous_scale='Reds', template="plotly_white")
    st.plotly_chart(fig_dept, use_container_width=True)

with row1_col2:
    st.subheader("Workforce Risk Profile (Active Only)")
    active_df = df[df['Status'] == 'Active']
    risk_counts = active_df['Risk_Score'].value_counts()
    fig_risk = px.pie(names=risk_counts.index, values=risk_counts.values, 
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_risk, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Experience vs Salary (Interactive Scatter)")
    # Sample data for scatter to keep it responsive
    sample_df = df.sample(min(2000, len(df)))
    fig_scatter = px.scatter(sample_df, x="Experience_Years", y="Salary_INR", 
                             color="Performance_Rating", size="Performance_Rating",
                             hover_data=['Job_Title', 'Department'], 
                             template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_col2:
    st.subheader("Salary Distribution (INR)")
    fig_hist = px.histogram(df, x="Salary_INR", nbins=50, 
                            color_discrete_sequence=['#00CC96'], template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

# --- RAW DATA VIEW ---
if st.checkbox("Show Sample High-Risk Employee Data"):
    st.write(df[df['Risk_Score'] == 'High (Flight Risk)'].head(50))
