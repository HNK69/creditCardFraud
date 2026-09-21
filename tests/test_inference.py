import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.inference.predict import predict_transaction


FEATURE_NAMES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


@pytest.fixture
def test_artifacts():
    """
    Create lightweight test-time artifacts.

    These are intentionally trained/generated only for CI testing.
    The real production model remains outside Git and is not used by
    the unit tests.
    """

    # Create a tiny synthetic training dataset with the same 30 features
    # expected by the production inference pipeline.
    X_train = pd.DataFrame(
        [
            [0.0] * 30,
            [1.0] * 30,
            [-1.0] * 30,
            [0.5] * 30,
        ],
        columns=FEATURE_NAMES,
    )

    # Binary labels are enough to fit a small test classifier.
    y_train = [0, 1, 0, 1]

    # Fit the same type of scaler used by production.
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Fit a tiny XGBoost model only for testing the inference contract.
    model = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )
    model.fit(scaler.transform(X_train), y_train)

    # Use a simple test threshold.
    threshold = 0.5

    return model, scaler, threshold


def valid_transaction():
    """Return a valid transaction containing all expected features."""

    return pd.DataFrame(
        [[0.0] * len(FEATURE_NAMES)],
        columns=FEATURE_NAMES,
    )


def test_prediction_returns_expected_fields(test_artifacts):
    """Verify that inference returns the expected response structure."""

    model, scaler, threshold = test_artifacts

    result = predict_transaction(
        valid_transaction(),
        model=model,
        scaler=scaler,
        threshold=threshold,
    )

    assert "fraud_probability" in result
    assert "prediction" in result
    assert "decision" in result


def test_prediction_probability_is_valid(test_artifacts):
    """Verify probability and prediction values are valid."""

    model, scaler, threshold = test_artifacts

    result = predict_transaction(
        valid_transaction(),
        model=model,
        scaler=scaler,
        threshold=threshold,
    )

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["prediction"] in [0, 1]
    assert result["decision"] in ["Fraud", "Legitimate"]


def test_missing_feature_is_rejected(test_artifacts):
    """Verify missing input features are rejected."""

    model, scaler, threshold = test_artifacts

    transaction = valid_transaction().drop(columns=["V1"])

    with pytest.raises(ValueError):
        predict_transaction(
            transaction,
            model=model,
            scaler=scaler,
            threshold=threshold,
        )


def test_extra_features_is_rejected(test_artifacts):
    """Verify unexpected input features are rejected."""

    model, scaler, threshold = test_artifacts

    transaction = valid_transaction()
    transaction["ExtraFeature"] = 0.8

    with pytest.raises(ValueError):
        predict_transaction(
            transaction,
            model=model,
            scaler=scaler,
            threshold=threshold,
        )


def test_non_numeric_feature_is_rejected(test_artifacts):
    """Verify non-numeric transaction values are rejected."""

    model, scaler, threshold = test_artifacts

    transaction = valid_transaction()
    transaction["Amount"] = "invalid"

    with pytest.raises(TypeError):
        predict_transaction(
            transaction,
            model=model,
            scaler=scaler,
            threshold=threshold,
        )


def test_missing_value_is_rejected(test_artifacts):
    """Verify missing transaction values are rejected."""

    model, scaler, threshold = test_artifacts

    transaction = valid_transaction()
    transaction.loc[0, "Amount"] = float("nan")

    with pytest.raises(ValueError):
        predict_transaction(
            transaction,
            model=model,
            scaler=scaler,
            threshold=threshold,
        )