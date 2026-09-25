import joblib
import pandas as pd

from pathlib import Path
from data_processing import clean_data


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"


REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


ALLOWED_VALUES = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}


def load_model():
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    threshold = bundle["threshold"]

    return model, threshold


def validate_customer_data(customer_data: dict):
    # 1) Check missing features
    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # 2) Check extra / unexpected features
    extra_features = [
        feature
        for feature in customer_data
        if feature not in REQUIRED_FEATURES
    ]

    if extra_features:
        raise ValueError(
            f"Unexpected features: {extra_features}"
        )

    # 3) Check categorical values
    for feature, allowed_values in ALLOWED_VALUES.items():
        if customer_data[feature] not in allowed_values:
            raise ValueError(
                f"Invalid value for {feature}: "
                f"{customer_data[feature]}. "
                f"Allowed values are: {allowed_values}"
            )

    # 4) Check numeric types
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for feature in numeric_features:
        if not isinstance(
            customer_data[feature],
            (int, float)
        ):
            raise TypeError(
                f"{feature} must be a number."
            )

    # 5) Check numeric ranges
    if customer_data["tenure"] < 0:
        raise ValueError(
            "tenure cannot be negative."
        )

    if customer_data["MonthlyCharges"] < 0:
        raise ValueError(
            "MonthlyCharges cannot be negative."
        )

    if customer_data["TotalCharges"] < 0:
        raise ValueError(
            "TotalCharges cannot be negative."
        )

    # 6) Business consistency for internet services
    if customer_data["InternetService"] == "No":
        internet_dependent_features = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        for feature in internet_dependent_features:
            if (
                customer_data[feature]
                != "No internet service"
            ):
                raise ValueError(
                    f"{feature} must be "
                    f"'No internet service' "
                    f"when InternetService is 'No'."
                )

    # 7) Business consistency for phone service
    if customer_data["PhoneService"] == "No":
        if (
            customer_data["MultipleLines"]
            != "No phone service"
        ):
            raise ValueError(
                "MultipleLines must be "
                "'No phone service' "
                "when PhoneService is 'No'."
            )


def predict_churn(customer_data: dict):
    validate_customer_data(customer_data)

    model, threshold = load_model()

    customer_df = pd.DataFrame(
        [customer_data]
    )

    customer_df = clean_data(
        customer_df
    )

    probability = model.predict_proba(
        customer_df
    )[:, 1][0]

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold
    }


if __name__ == "__main__":
    customer = {
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
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0
    }

    result = predict_churn(
        customer
    )

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Probability:",
        result["probability"]
    )

    print(
        "Threshold:",
        result["threshold"]
    )