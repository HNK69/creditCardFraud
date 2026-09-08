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

    if transaction.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Expected {scaler.n_features_in_} features, "
            f"but received {transaction.shape[1]}."
        )

    scaled_transaction = scaler.transform(transaction)

    fraud_probability = model.predict_proba(scaled_transaction)[:, 1][0]

    prediction = int(fraud_probability >= threshold)

    return{
        "fraud_probability": float(fraud_probability),
        "prediction": prediction,
        "decision": "Fraud" if prediction == 1 else "Legitimate",
    }