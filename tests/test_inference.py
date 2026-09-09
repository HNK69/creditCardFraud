import pandas as pd
from src.inference.predict import predict_transaction


def test_prediction_returns_expected_fields():

    feature_names = [
        "Time",
        *[f"V{i}" for i in range(1,29)],
        "Amount",
    ]

    transaction = pd.DataFrame(
        [[0.0] * len(feature_names)],
        columns = feature_names,
    )

    result = predict_transaction(transaction)

    assert "fraud_probability" in result
    assert "prediction" in result
    assert "decision" in result

def test_prediction_probability_is_valid():

    feature_names = [
        "Time",
        *[f"V{i}" for i in range(1,29)],
        "Amount",
    ]

    transaction = pd.DataFrame(
        [[0.0] * len(feature_names)],
        columns = "feature_names",
    )

    result = predict_transaction(transaction)

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["prediction"] in [0,1]
    assert result["decision"] in ["Fraud", "Legitimate"]
