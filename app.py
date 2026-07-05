"""
Real-time Telco Churn prediction demo.
Loads the trained pipeline (preprocessing + scaling + Logistic Regression,
all bundled into one .joblib file) and scores a customer instantly.

Run locally:
    streamlit run app.py
Deploy: push this repo to GitHub, then point Streamlit Community Cloud
at app.py (see README.md).
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# IMPORTANT: the pipeline was pickled with a FunctionTransformer that
# references a module literally named "preprocessing" (see src/preprocessing.py).
# joblib/pickle re-imports that exact module path when loading, so "src"
# must be on sys.path BEFORE joblib.load() runs, even though app.py
# itself lives at the repo root.
sys.path.append(str(Path(__file__).parent / "src"))

MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.title("📉 Telco Customer Churn Predictor")
st.caption(
    "Tuned Logistic Regression — 80% recall on churners, 0.84 ROC-AUC on a held-out test set. "
    "Enter a customer profile below to get a live churn risk score."
)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
    }])

    proba = pipeline.predict_proba(row)[0][1]
    pred = pipeline.predict(row)[0]

    st.divider()
    st.metric("Churn Probability", f"{proba:.1%}")

    if pred == 1:
        st.error("⚠️ High Risk — this customer is likely to churn.")
    else:
        st.success("✅ Low Risk — this customer is likely to stay.")

    st.progress(min(max(proba, 0.0), 1.0))

    with st.expander("What drives this?"):
        st.write(
            "Month-to-month contracts, low tenure, high monthly charges, and paying by "
            "electronic check are the strongest churn signals in this model. Long contracts "
            "and tech support / online security add-ons reduce risk."
        )

st.divider()
st.caption("Model trained on the IBM/Kaggle Telco Customer Churn dataset (7,043 customers).")
