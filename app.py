# app.py
# --------------------------------------------------
# Employee Attrition Analysis - HR Analytics Dashboard
# (PIPELINE SAFE | POWER BI STYLE)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load PIPELINE (ONLY ONE ARTIFACT)
# -----------------------------
pipeline = joblib.load("models/attrition_pipeline.pkl")

raw_df = pd.read_csv("data/processed/cleaned_hr_data.csv")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

# -----------------------------
# Power BI–style UI
# -----------------------------
st.markdown("""
<style>
h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.title("👥 Employee Attrition Analysis – HR Dashboard")

# ==================================================
# TABS (Power BI Pages)
# ==================================================
tab1, tab2, tab3 = st.tabs(
    ["📊 HR Overview", "🧾 Prediction", "📈 HR Analytics"]
)

# ==================================================
# TAB 1: KPI SECTION
# ==================================================
with tab1:
    st.subheader("📊 HR KPIs")

    total_employees = raw_df.shape[0]
    attrition_count = raw_df["Attrition"].sum()
    attrition_rate = (attrition_count / total_employees) * 100
    avg_tenure = raw_df["YearsAtCompany"].mean()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Employees", total_employees)
    kpi2.metric("Attrition Count", attrition_count)
    kpi3.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
    kpi4.metric("Avg Tenure (Years)", f"{avg_tenure:.1f}")

    st.info(
        "📌 Higher attrition is observed among employees with lower tenure "
        "and specific departments & job roles."
    )

# ==================================================
# TAB 2: PREDICTION UI (NO FEATURE MISMATCH)
# ==================================================
with tab2:
    st.subheader("🧾 Employee Attrition Prediction")

    age = st.slider("Age", 18, 60, 30)

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=1000
    )

    years_at_company = st.slider("Years at Company", 0, 40, 3)

    department = st.selectbox(
        "Department",
        raw_df["Department"].unique()
    )

    job_role = st.selectbox(
        "Job Role",
        raw_df["JobRole"].unique()
    )

    business_travel = st.selectbox(
        "Business Travel",
        raw_df["BusinessTravel"].unique()
    )

    education_field = st.selectbox(
        "Education Field",
        raw_df["EducationField"].unique()
    )

    gender = st.selectbox(
        "Gender",
        raw_df["Gender"].unique()
    )

    marital_status = st.selectbox(
        "Marital Status",
        raw_df["MaritalStatus"].unique()
    )

    overtime = st.selectbox(
        "OverTime",
        raw_df["OverTime"].unique()
    )

    work_life_balance = st.slider("Work Life Balance (1–4)", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)

    # --------------------------------------------------
    # RAW FEATURE ROW (PIPELINE HANDLES EVERYTHING)
    # --------------------------------------------------
    input_df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "WorkLifeBalance": work_life_balance,
        "JobSatisfaction": job_satisfaction,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime
    }])

    if st.button("🔍 Predict Attrition"):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        st.subheader("📌 Prediction Result")
        st.progress(probability)
        st.write(f"**Attrition Probability:** `{probability:.2f}`")

        if probability < 0.4:
            st.success("🟢 Low Attrition Risk")
        elif probability < 0.7:
            st.warning("🟡 Medium Attrition Risk")
        else:
            st.error("🔴 High Attrition Risk")

# ==================================================
# TAB 3: HR ANALYTICS (POWER BI STYLE GRAPHS)
# ==================================================
with tab3:
    st.subheader("📈 HR Analytics Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Attrition by Department**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Department", hue="Attrition", data=raw_df, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Attrition by Job Role**")
        fig2, ax2 = plt.subplots()
        sns.countplot(y="JobRole", hue="Attrition", data=raw_df, ax=ax2)
        st.pyplot(fig2)

    st.markdown("**Age vs Attrition**")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Attrition", y="Age", data=raw_df, ax=ax3)
    st.pyplot(fig3)

    st.markdown("**Monthly Income vs Attrition**")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="Attrition", y="MonthlyIncome", data=raw_df, ax=ax4)
    st.pyplot(fig4)

    st.info(
        "📌 Employees with lower income, lower tenure, and specific job roles "
        "show higher attrition probability."
    )

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Employee Attrition Analysis | HR Analytics Dashboard | Internship Project")
