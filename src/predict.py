from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_processing import clean_data


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_model.joblib"
)


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
    "TotalCharges",
]


ALLOWED_VALUES = {
    "gender": [
        "Male",
        "Female",
    ],
    "SeniorCitizen": [
        0,
        1,
    ],
    "Partner": [
        "Yes",
        "No",
    ],
    "Dependents": [
        "Yes",
        "No",
    ],
    "PhoneService": [
        "Yes",
        "No",
    ],
    "MultipleLines": [
        "Yes",
        "No",
        "No phone service",
    ],
    "InternetService": [
        "DSL",
        "Fiber optic",
        "No",
    ],
    "OnlineSecurity": [
        "Yes",
        "No",
        "No internet service",
    ],
    "OnlineBackup": [
        "Yes",
        "No",
        "No internet service",
    ],
    "DeviceProtection": [
        "Yes",
        "No",
        "No internet service",
    ],
    "TechSupport": [
        "Yes",
        "No",
        "No internet service",
    ],
    "StreamingTV": [
        "Yes",
        "No",
        "No internet service",
    ],
    "StreamingMovies": [
        "Yes",
        "No",
        "No internet service",
    ],
    "Contract": [
        "Month-to-month",
        "One year",
        "Two year",
    ],
    "PaperlessBilling": [
        "Yes",
        "No",
    ],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# =========================================================
# Model Loading
# =========================================================

def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    if (
        "model" not in bundle
        or "threshold" not in bundle
    ):
        raise ValueError(
            "Invalid model bundle. "
            "Expected 'model' and 'threshold'."
        )

    return bundle["model"], float(
        bundle["threshold"]
    )


MODEL, THRESHOLD = load_model_bundle()


# =========================================================
# Input Validation
# =========================================================

def validate_required_features(
    customer_data: dict[str, Any],
) -> None:
    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: "
            f"{missing_features}"
        )

    extra_features = [
        feature
        for feature in customer_data
        if feature not in REQUIRED_FEATURES
    ]

    if extra_features:
        raise ValueError(
            f"Unexpected features: "
            f"{extra_features}"
        )


def validate_categorical_values(
    customer_data: dict[str, Any],
) -> None:
    for (
        feature,
        allowed_values,
    ) in ALLOWED_VALUES.items():
        if (
            customer_data[feature]
            not in allowed_values
        ):
            raise ValueError(
                f"Invalid value for {feature}: "
                f"{customer_data[feature]}. "
                f"Allowed values are: "
                f"{allowed_values}"
            )


def validate_numeric_values(
    customer_data: dict[str, Any],
) -> None:
    for feature in NUMERIC_FEATURES:
        value = customer_data[feature]

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{feature} must be a number."
            )

        if value < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )


def validate_business_rules(
    customer_data: dict[str, Any],
) -> None:
    if (
        customer_data["InternetService"]
        == "No"
    ):
        internet_dependent_features = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        for feature in (
            internet_dependent_features
        ):
            if (
                customer_data[feature]
                != "No internet service"
            ):
                raise ValueError(
                    f"{feature} must be "
                    f"'No internet service' "
                    f"when InternetService "
                    f"is 'No'."
                )

    if (
        customer_data["PhoneService"]
        == "No"
        and customer_data["MultipleLines"]
        != "No phone service"
    ):
        raise ValueError(
            "MultipleLines must be "
            "'No phone service' "
            "when PhoneService is 'No'."
        )


def validate_customer_data(
    customer_data: dict[str, Any],
) -> None:
    if not isinstance(
        customer_data,
        dict,
    ):
        raise TypeError(
            "customer_data must be a dictionary."
        )

    validate_required_features(
        customer_data
    )

    validate_categorical_values(
        customer_data
    )

    validate_numeric_values(
        customer_data
    )

    validate_business_rules(
        customer_data
    )


# =========================================================
# Prediction
# =========================================================

def predict_churn(
    customer_data: dict[str, Any],
) -> dict[str, Any]:
    validate_customer_data(
        customer_data
    )

    customer_df = pd.DataFrame(
        [customer_data]
    )

    customer_df = clean_data(
        customer_df
    )

    probability = float(
        MODEL.predict_proba(
            customer_df
        )[:, 1][0]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    label = (
        "Churn"
        if prediction == 1
        else "No Churn"
    )

    return {
        "prediction": prediction,
        "label": label,
        "probability": probability,
        "threshold": THRESHOLD,
    }


# =========================================================
# Manual Smoke Test
# =========================================================

def main():
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
        "TotalCharges": 190.0,
    }

    result = predict_churn(
        customer
    )

    print("=" * 50)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 50)

    print(
        "Prediction:",
        result["prediction"],
    )

    print(
        "Label:",
        result["label"],
    )

    print(
        "Probability:",
        f"{result['probability']:.4f}",
    )

    print(
        "Threshold:",
        result["threshold"],
    )


if __name__ == "__main__":
    main()