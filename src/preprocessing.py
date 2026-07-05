"""
Shared preprocessing logic for the Telco Churn model.

This module is imported by BOTH train_model.py and app.py so that
training-time and inference-time feature engineering can never drift
apart. It replaces the ad-hoc, cell-by-cell mapping in the original
notebook with a single deterministic function that can be wrapped in
a scikit-learn Pipeline and shipped inside one .joblib file.
"""

import pandas as pd

# Columns that are simple Yes/No -> 1/0
BINARY_YES_NO = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Columns with a "No internet/phone service" tier, encoded 0/1/2
TIERED_MAPS = {
    "MultipleLines": {"No phone service": 0, "No": 1, "Yes": 2},
    "OnlineSecurity": {"No internet service": 0, "No": 1, "Yes": 2},
    "OnlineBackup": {"No internet service": 0, "No": 1, "Yes": 2},
    "DeviceProtection": {"No internet service": 0, "No": 1, "Yes": 2},
    "TechSupport": {"No internet service": 0, "No": 1, "Yes": 2},
}

# Matches sklearn's LabelEncoder default (alphabetical) ordering that
# the original notebook relied on: Month-to-month < One year < Two year
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}

PAYMENT_MAP = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}

# Final feature order the model was trained on.
# NOTE: InternetService, StreamingTV, StreamingMovies are intentionally
# excluded (dropped in the original notebook after correlation review),
# so the app never needs to ask for them.
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges"]


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Take a dataframe with the ORIGINAL raw column values (as they
    appear in the CSV / a web form) and return the numeric feature
    frame the model expects. Safe to call on a single-row dataframe
    at inference time or the full training set.
    """
    df = df.copy()

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    for col in BINARY_YES_NO:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    for col, mapping in TIERED_MAPS.items():
        df[col] = df[col].map(mapping)

    df["Contract"] = df["Contract"].map(CONTRACT_MAP)
    df["PaymentMethod"] = df["PaymentMethod"].map(PAYMENT_MAP)

    missing = df[FEATURE_ORDER].isnull().any()
    if missing.any():
        bad_cols = missing[missing].index.tolist()
        raise ValueError(
            f"Unmapped/unknown category values in columns: {bad_cols}. "
            "Check the input against the categories this model was trained on."
        )

    return df[FEATURE_ORDER]
