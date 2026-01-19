# src/model_training.py
# --------------------------------------------------
# Employee Attrition Analysis - Model Training
# (PIPELINE SAFE + MODEL COMPARISON)
# --------------------------------------------------

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# File paths
# -----------------------------
DATA_PATH = "data/processed/cleaned_hr_data.csv"
MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "attrition_pipeline.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Target and features
y = df["Attrition"]
X = df.drop("Attrition", axis=1)

# -----------------------------
# Column groups
# -----------------------------
numeric_cols = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "WorkLifeBalance",
    "JobSatisfaction"
]

categorical_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

# -----------------------------
# Preprocessing pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================================
# 1️⃣ Logistic Regression Pipeline
# ==================================================
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

lr_pipeline.fit(X_train, y_train)

lr_preds = lr_pipeline.predict(X_test)
lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_recall = recall_score(y_test, lr_preds)
lr_roc = roc_auc_score(y_test, lr_probs)

print("\n🔹 Logistic Regression Results")
print("Accuracy:", lr_accuracy)
print("Recall:", lr_recall)
print("ROC-AUC:", lr_roc)
print(confusion_matrix(y_test, lr_preds))
print(classification_report(y_test, lr_preds))

# ==================================================
# 2️⃣ Random Forest Pipeline
# ==================================================
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_preds)
rf_recall = recall_score(y_test, rf_preds)
rf_roc = roc_auc_score(y_test, rf_probs)

print("\n🔹 Random Forest Results")
print("Accuracy:", rf_accuracy)
print("Recall:", rf_recall)
print("ROC-AUC:", rf_roc)
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# ==================================================
# Save BEST pipeline (based on ROC-AUC)
# ==================================================
best_pipeline = rf_pipeline if rf_roc > lr_roc else lr_pipeline

joblib.dump(best_pipeline, PIPELINE_PATH)

print("\n✅ Best pipeline saved at:", PIPELINE_PATH)
print("✅ Training completed successfully")
