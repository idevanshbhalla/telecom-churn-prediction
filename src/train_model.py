"""
Trains the champion churn model (tuned Logistic Regression, matching
the notebook's own conclusion) and saves ONE joblib file that contains
preprocessing + scaling + the model together.

Run from the repo root:
    python src/train_model.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.append(str(Path(__file__).parent))
from preprocessing import FEATURE_ORDER, NUMERIC_COLS, preprocess_raw  # noqa: E402

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_pipeline.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X_raw = df.drop(columns=["customerID", "TotalCharges", "Churn"])

    # Split BEFORE anything else touches the data -> no test-set peeking.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("clean", FunctionTransformer(preprocess_raw)),
        ("scale", ColumnTransformer(
            transformers=[("num", StandardScaler(), NUMERIC_COLS)],
            remainder="passthrough",
        )),
        ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])

    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["liblinear", "lbfgs"],
    }

    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train_raw, y_train)

    best_pipeline = grid.best_estimator_
    print(f"Best params: {grid.best_params_}\n")

    y_pred = best_pipeline.predict(X_test_raw)
    y_proba = best_pipeline.predict_proba(X_test_raw)[:, 1]

    print("--- Holdout Test Set Performance ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline -> {MODEL_PATH}")
    print(f"Feature order expected at inference: {FEATURE_ORDER}")


if __name__ == "__main__":
    main()
