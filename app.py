# ==========================================
# JOB ACCEPTANCE PREDICTION DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    layout="wide"
)

st.title("🎯 Job Acceptance Prediction System")
st.markdown("HR Analytics Dashboard for Offer Acceptance Insights")

# ==========================================
# LOAD DATA & MODEL
# ==========================================


import pandas as pd
import joblib

df = pd.read_csv("data/final/feature_engineered_encoded.csv")
model = joblib.load("models/job_acceptance_model.pkl")



# ==========================================
# KPI CALCULATIONS
# ==========================================

total_candidates = len(df)
accepted = df['status'].sum()
rejected = total_candidates - accepted

acceptance_rate = (accepted / total_candidates) * 100
dropout_rate = (rejected / total_candidates) * 100

avg_interview_score = df['interview_total_score'].mean()
avg_skills_match = df['skills_match_percentage'].mean()

high_risk_candidates = df['offer_risk'].sum()
high_risk_percent = (high_risk_candidates / total_candidates) * 100

# ==========================================
# KPI DISPLAY
# ==========================================

st.subheader("📌 Key HR Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Candidates", total_candidates)
col2.metric("Acceptance Rate (%)", f"{acceptance_rate:.2f}")
col3.metric("Offer Dropout Rate (%)", f"{dropout_rate:.2f}")
col4.metric("Avg Interview Score", f"{avg_interview_score:.2f}")
col5.metric("High Risk (%)", f"{high_risk_percent:.2f}")

# ==========================================
# DATA OVERVIEW
# ==========================================

st.subheader("📊 Candidate Overview")
st.dataframe(df.head(50))

# ==========================================
# ACCEPTANCE DISTRIBUTION
# ==========================================

st.subheader("📈 Job Acceptance Distribution")

status_count = df['status'].value_counts().rename({1: "Accepted", 0: "Rejected"})
st.bar_chart(status_count)

# ==========================================
# FEATURE IMPORTANCE VIEW
# ==========================================

st.subheader("🧠 Key Factors Influencing Acceptance")

importance = model.feature_importances_
features = df.drop('status', axis=1).columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(10)

st.dataframe(importance_df)

# ==========================================
# PREDICTION SECTION
# ==========================================

st.subheader("🔮 Predict Job Acceptance for Random Candidate")

if st.button("Predict Random Candidate"):

    # Randomly select one candidate
    sample_input = df.sample(1)

    st.write("### 📋 Selected Candidate Details:")
    st.dataframe(sample_input)

    # Separate features from target
    X_input = sample_input.drop("status", axis=1)

    # Predict
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    # Show result
    if prediction == 1:
        st.success("✅ Candidate is likely to ACCEPT the job offer")
    else:
        st.error("❌ Candidate is likely to REJECT the job offer")

    # Show probability
    st.write("📊 Acceptance Probability:", round(proba[1]*100, 2), "%")
    st.write("📊 Rejection Probability:", round(proba[0]*100, 2), "%")




# ==========================================
# END OF DASHBOARD
# ==========================================
