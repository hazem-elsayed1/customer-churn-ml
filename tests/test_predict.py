import pytest

from src.predict import validate_customer_data


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