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

    expected_features = list(scaler.feature_name_in_)
    received_features = list(transaction.columns)

    missing_features = set(expected_features) - set(received_features)
    extra_features = set(received_features) - set(expected_features)

    if missing_features or extra_features:
        raise ValueError(
            f"Invalid input columns. "
            f"Missing: {sorted(missing_features)}; "
            f"Extra: {sorted(extra_features)}"
        )

    