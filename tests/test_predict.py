import pytest
import numpy as np
from src.predict import (
    load_model_bundle,
    predict_churn,
    validate_customer_data,
)


def valid_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 179.0,
    }


def test_valid_customer_passes_validation():
    customer = valid_customer()

    validate_customer_data(customer)


def test_missing_feature_raises_error():
    customer = valid_customer()
    customer.pop("tenure")

    with pytest.raises(ValueError):
        validate_customer_data(customer)


def test_extra_feature_raises_error():
    customer = valid_customer()
    customer["unexpected_feature"] = "test"

    with pytest.raises(ValueError):
        validate_customer_data(customer)


def test_negative_numeric_value_raises_error():
    customer = valid_customer()
    customer["tenure"] = -1

    with pytest.raises(ValueError):
        validate_customer_data(customer)


def test_invalid_category_raises_error():
    customer = valid_customer()
    customer["Contract"] = "Five years"

    with pytest.raises(ValueError):
        validate_customer_data(customer)


def test_no_internet_requires_no_internet_service_values():
    customer = valid_customer()
    customer["InternetService"] = "No"
    customer["OnlineSecurity"] = "No"

    with pytest.raises(ValueError):
        validate_customer_data(customer)


def test_no_phone_requires_no_phone_service_value():
    customer = valid_customer()
    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "No"

    with pytest.raises(ValueError):
        validate_customer_data(customer)

def test_nan_numeric_value_raises_error():
    customer = valid_customer()
    customer["MonthlyCharges"] = float("nan")

    with pytest.raises(
        ValueError,
        match="must be a finite number",
    ):
        validate_customer_data(customer)


def test_infinite_numeric_value_raises_error():
    customer = valid_customer()
    customer["tenure"] = float("inf")

    with pytest.raises(
        ValueError,
        match="must be a finite number",
    ):
        validate_customer_data(customer)


def test_boolean_numeric_value_raises_error():
    customer = valid_customer()
    customer["tenure"] = True

    with pytest.raises(
        TypeError,
        match="not a boolean",
    ):
        validate_customer_data(customer)

class StubModel:
    def predict_proba(self, X):
        return np.array(
            [
                [0.25, 0.75],
            ]
        )

def test_predict_churn_uses_model_and_threshold(monkeypatch):
    customer = valid_customer()

    monkeypatch.setattr(
        "src.predict.load_model_bundle",
        lambda: (StubModel(), 0.60),
    )

    result = predict_churn(customer)

    assert result["prediction"] == 1
    assert result["label"] == "Churn"
    assert result["probability"] == 0.75
    assert result["threshold"] == 0.60


def test_load_model_bundle_missing_file_raises_error(
    monkeypatch,
    tmp_path,
):
    fake_model_path = tmp_path / "missing_model.joblib"

    monkeypatch.setattr(
        "src.predict.MODEL_PATH",
        fake_model_path,
    )

    load_model_bundle.cache_clear()

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        load_model_bundle()

    load_model_bundle.cache_clear()


def test_load_model_bundle_invalid_bundle_raises_error(
    monkeypatch,
    tmp_path,
):
    fake_model_path = tmp_path / "invalid_model.joblib"
    fake_model_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "src.predict.MODEL_PATH",
        fake_model_path,
    )

    monkeypatch.setattr(
        "src.predict.joblib.load",
        lambda path: {"wrong_key": "value"},
    )

    load_model_bundle.cache_clear()

    with pytest.raises(
        ValueError,
        match="Invalid model bundle",
    ):
        load_model_bundle()

    load_model_bundle.cache_clear()