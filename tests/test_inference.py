import pandas as pd
from src.inference.predict import predict_transaction
import pytest 

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
        columns = feature_names,
    )

    result = predict_transaction(transaction)

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["prediction"] in [0,1]
    assert result["decision"] in ["Fraud", "Legitimate"]


def valid_transaction():

    feature_names = [
        "Time",
        *[f"V{i}" for i in range(1,29)],
        "Amount",
    ]

    return pd.DataFrame(
        [[0.0] * len(feature_names)],
        columns = feature_names,
    )

def test_missing_feature_is_rejected():

    transaction = valid_transaction().drop(columns = ["V1"])

    with pytest.raises(ValueError):
        predict_transaction(transaction)


def test_extra_features_is_rejected():

    transaction = valid_transaction()
    transaction["ExtraFeature"] = 0.8

    with pytest.raises(ValueError):
        predict_transaction(transaction)


def test_non_numeric_feature_is_rejected():

    transaction = valid_transaction()
    transaction["Amount"] = "invalid"

    with pytest.raises(TypeError):
        predict_transaction(transaction)

def test_missing_value_is_rejected():

    transaction = valid_transaction()
    transaction.loc[0, "Amount"] = float("nan")

    with pytest.raises(ValueError):
        predict_transaction(transaction)