# src/data_preprocessing.py
# ---------------------------------------------
# Employee Attrition Analysis - Data Preprocessing
# ---------------------------------------------

import pandas as pd
import os

# -----------------------------
# File Paths
# -----------------------------
RAW_DATA_PATH = "data/raw/hr_attrition.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_hr_data.csv"

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"Dataset shape before cleaning: {df.shape}")

# -----------------------------
# Drop unnecessary columns
# -----------------------------
# These columns do not add value for analytics or prediction
drop_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df.drop(columns=drop_columns, inplace=True)

# -----------------------------
# Encode target variable
# -----------------------------
# Attrition: Yes -> 1, No -> 0
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# -----------------------------
# Check missing values
# -----------------------------
if df.isnull().sum().any():
    print("Missing values found. Filling with median/mode.")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
else:
    print("No missing values found.")

# -----------------------------
# Save cleaned dataset
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)

print("Data preprocessing completed successfully.")
print(f"Cleaned dataset saved at: {PROCESSED_DATA_PATH}")
print(f"Final dataset shape: {df.shape}")
