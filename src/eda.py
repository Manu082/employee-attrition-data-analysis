# src/eda.py
# ---------------------------------------------
# Employee Attrition Analysis - Exploratory Data Analysis
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load cleaned dataset
# -----------------------------
DATA_PATH = "data/processed/cleaned_hr_data.csv"
df = pd.read_csv(DATA_PATH)

print("EDA started...")
print(df.head())

# -----------------------------
# Set visualization style
# -----------------------------
sns.set(style="whitegrid")

# -----------------------------
# 1. Attrition Distribution
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition (0 = No, 1 = Yes)")
plt.ylabel("Employee Count")
plt.show()

# -----------------------------
# 2. Attrition by Department
# -----------------------------
plt.figure(figsize=(8, 4))
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Attrition by Department")
plt.xticks(rotation=15)
plt.show()

# -----------------------------
# 3. Attrition by Job Role
# -----------------------------
plt.figure(figsize=(10, 5))
sns.countplot(y="JobRole", hue="Attrition", data=df)
plt.title("Attrition by Job Role")
plt.show()

# -----------------------------
# 4. Age vs Attrition
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Attrition", y="Age", data=df)
plt.title("Age vs Attrition")
plt.show()

# -----------------------------
# 5. Monthly Income vs Attrition
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")
plt.show()

# -----------------------------
# 6. Years at Company vs Attrition
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Attrition", y="YearsAtCompany", data=df)
plt.title("Years at Company vs Attrition")
plt.show()

print("EDA completed successfully.")
