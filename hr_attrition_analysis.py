import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("hr_data.csv")

# 1. ATTRITION RATE ANALYSIS
print("Calculating Attrition Rates...")

# Define attrition: Resigned or Terminated
df['Is_Attrition'] = df['Status'].apply(lambda x: 1 if x in ['Resigned', 'Terminated'] else 0)

# Attrition Rate by Department
dept_attrition = df.groupby('Department')['Is_Attrition'].mean() * 100
dept_attrition = dept_attrition.sort_values(ascending=False)

# Attrition Rate by Work Mode
mode_attrition = df.groupby('Work_Mode')['Is_Attrition'].mean() * 100

# 2. HEURISTIC PREDICTIVE MODEL (Risk Scoring)
print("Running Risk Factor Analysis...")

# Calculate Median Salary per Job Title to identify underpaid high-performers
median_salary_role = df.groupby('Job_Title')['Salary_INR'].transform('median')
df['Relative_Salary'] = df['Salary_INR'] / median_salary_role

# Risk Heuristics:
# - Underpaid High Performer: Rating >= 4 and Salary < Median (Risk of Resigning)
# - Overpaid Low Performer: Rating <= 2 and Salary > Median (Risk of Termination)

df['Risk_Score'] = 'Low'
df.loc[(df['Performance_Rating'] >= 4) & (df['Relative_Salary'] < 0.8), 'Risk_Score'] = 'High (Flight Risk)'
df.loc[(df['Performance_Rating'] <= 2) & (df['Relative_Salary'] > 1.2), 'Risk_Score'] = 'High (Performance Gap)'

risk_counts = df[df['Status'] == 'Active']['Risk_Score'].value_counts()

# 3. VISUALIZATION
print("Generating Plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Attrition Rate by Dept
sns.barplot(x=dept_attrition.index, y=dept_attrition.values, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Attrition Rate (%) by Department')
axes[0, 0].set_ylabel('Attrition %')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Attrition Rate by Work Mode
sns.barplot(x=mode_attrition.index, y=mode_attrition.values, ax=axes[0, 1], palette='magma')
axes[0, 1].set_title('Attrition Rate (%) by Work Mode')
axes[0, 1].set_ylabel('Attrition %')

# Plot 3: Risk Score Distribution (Active Employees)
axes[1, 0].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#ffcc99'])
axes[1, 0].set_title('Risk Profile of Active Workforce')

# Plot 4: Performance vs Salary Scatter (Sampled)
sns.scatterplot(data=df.sample(1000), x='Experience_Years', y='Salary_INR', hue='Performance_Rating', ax=axes[1, 1], alpha=0.6)
axes[1, 1].set_title('Salary vs Experience (Sample of 1000)')

plt.tight_layout()
plt.savefig('attrition_analysis_report.png')
print("Analysis complete. Report saved as 'attrition_analysis_report.png'.")

# Output key statistics
print("\n--- Key Statistics ---")
print(f"Overall Attrition Rate: {df['Is_Attrition'].mean()*100:.2f}%")
print("\nTop 3 Departments by Attrition:")
print(dept_attrition.head(3))
print("\nRisk Profiling (Active Employees):")
print(risk_counts)
