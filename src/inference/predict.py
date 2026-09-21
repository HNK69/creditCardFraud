import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "final_xgb_model.joblib"
SCALER_PATH = PROJECT_ROOT/ "models"/ "standard_scaler.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "fraud_threshold.joblib"

def _load_artifacts():
    """
    Load the production model artifacts only when they are actually needed.

    This keeps production inference connected to the real trained model,
    while allowing unit tests to inject lightweight test artifacts without
    requiring production model files in the Git repository.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, scaler, threshold

def predict_transaction(
    transaction: pd.DataFrame,
    model=None,
    scaler=None,
    threshold=None,
) -> dict:
    """
    Predict whether a transaction is fraudulent.

    In production, the real model artifacts are loaded automatically.
    During testing, model, scaler, and threshold can be injected so that
    tests do not depend on production model files.
    """

    # Load the real production artifacts only when they were not supplied.
    if model is None or scaler is None or threshold is None:
        model, scaler, threshold = _load_artifacts()

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