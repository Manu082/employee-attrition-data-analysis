# src/feature_engineering.py
# --------------------------------------------------
# Employee Attrition Analysis - Feature Engineering
# --------------------------------------------------

import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# File paths
# -----------------------------
INPUT_DATA_PATH = "data/processed/cleaned_hr_data.csv"
OUTPUT_DATA_PATH = "data/processed/engineered_hr_data.csv"
SCALER_PATH = "models/scaler.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv(INPUT_DATA_PATH)
print("Loaded cleaned HR data")

# -----------------------------
# Separate target variable
# -----------------------------
y = df["Attrition"]
X = df.drop("Attrition", axis=1)

# -----------------------------
# Identify categorical columns
# -----------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# -----------------------------
# Encode categorical columns
# -----------------------------
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -----------------------------
# Scale numerical features
# -----------------------------
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, SCALER_PATH)
joblib.dump(X.columns.tolist(), FEATURE_NAMES_PATH)

# -----------------------------
# Save engineered dataset
# -----------------------------
final_df = pd.concat([X, y], axis=1)
final_df.to_csv(OUTPUT_DATA_PATH, index=False)

print("Feature engineering completed successfully.")
print(f"Engineered data saved at: {OUTPUT_DATA_PATH}")
print(f"Scaler saved at: {SCALER_PATH}")
print(f"Feature names saved at: {FEATURE_NAMES_PATH}")
