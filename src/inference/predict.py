import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "final_xgb_model.joblib"
SCALER_PATH = PROJECT_ROOT/ "models"/ "standard_scaler.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "fraud_threshold.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

def predict_transaction(transaction: pd.DataFrame) -> dict:

    expected_features = list(scaler.feature_names_in_)
    received_features = list(transaction.columns)

    missing_features = set(expected_features) - set(received_features)
    extra_features = set(received_features) - set(expected_features)

    if missing_features or extra_features:
        raise ValueError(
            f"Invalid input columns. "
            f"Missing: {sorted(missing_features)}; "
            f"Extra: {sorted(extra_features)}"
        )

    transaction = transaction[expected_features]

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in transaction.dtypes):
        raise TypeError("All transaction features must be numeric. ")

    if transaction.isnull().any().any():
        raise ValueError("Transaction contains missing values. ")

    if not transaction.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all().all():
        raise ValueError("Transaction contains infinite values. ")

    scaled_transaction = scaler.transform(transaction)

    fraud_probability = model.predict_proba(
        scaled_transaction
    )[:, 1][0]

    prediction = int(fraud_probability >= threshold)

    return{
        "fraud_probability": float(fraud_probability),
        "prediction": prediction,
        "decision": "Fraud" if prediction == 1 else "Legitimate"
    }