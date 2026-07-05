# Telco Customer Churn Prediction

Predicts whether a telecom customer will churn, using the IBM/Kaggle Telco
Customer Churn dataset (7,043 customers, 21 attributes). Five classifiers
were compared; a tuned Logistic Regression was selected as the champion
model and shipped as a single deployable pipeline.

**Live demo:** (https://telecom-churn-prediction-devansh.streamlit.app/)

## Results (held-out 30% test set)

| Metric | Score |
|---|---|
| Recall (Churn class) | 0.80 |
| ROC-AUC | 0.842 |
| Precision (Churn class) | 0.51 |
| Accuracy | 0.74 |

Recall was prioritized over raw accuracy: in churn prediction, the cost of
missing a customer who *will* leave (false negative) is higher than the
cost of a false alarm, since false alarms just trigger an unnecessary
retention offer.

## Key churn drivers

1. **Contract type** — month-to-month customers churn far more than
   1- or 2-year contract holders.
2. **Tenure** — risk is highest in a customer's first few months.
3. **Monthly charges** — higher bills correlate with higher churn.
4. **Payment method / add-ons** — electronic check payers and customers
   without tech support or online security churn more.

## Business recommendations

- Incentivize upgrades from month-to-month to annual contracts.
- Run a proactive check-in for customers in months 1–6.
- Flag high-monthly-charge + high-risk customers for a retention offer
  before they cancel.

## Repo structure

```
.
├── app.py                          # Streamlit real-time prediction demo
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_eda_and_modeling.ipynb   # Full EDA + 5-model comparison + tuning
├── src/
│   ├── preprocessing.py            # Single source of truth for feature engineering
│   └── train_model.py              # Trains and saves the deployable pipeline
└── models/
    └── churn_pipeline.joblib       # Preprocessing + scaling + model, bundled
```

The notebook is for analysis and storytelling — it runs top to bottom
without errors and shows the full model comparison. `src/` is the
production path: `preprocessing.py` is imported by both `train_model.py`
and `app.py`, so training-time and inference-time feature engineering
can never drift apart.

## Running it yourself

```bash
git clone <your-repo-url>
cd telco-churn-prediction
pip install -r requirements.txt

# Retrain from scratch (optional — a trained model is already committed)
python src/train_model.py

# Launch the real-time prediction app
streamlit run app.py
```

## Using the model programmatically

```python
import sys
sys.path.append("src")   # required so the pickled pipeline can resolve preprocessing.py

import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.joblib")

customer = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0,
}])

print(pipeline.predict_proba(customer)[0][1])  # churn probability
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, Streamlit, joblib.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample dataset, redistributed under Kaggle's terms).
