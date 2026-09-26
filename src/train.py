from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import clean_data


# =========================================================
# Configuration
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_model.joblib"
)


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]


# =========================================================
# Data Loading and Preparation
# =========================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df = clean_data(df)

    return df


def prepare_features(
    df: pd.DataFrame,
):
    X = df.drop(
        columns=[
            "Churn",
            "customerID",
        ]
    )

    y = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# =========================================================
# Preprocessing
# =========================================================

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# =========================================================
# Cross-Validation Setup
# =========================================================

def build_cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# =========================================================
# Candidate Model Comparison
# =========================================================

def compare_models(
    preprocessor,
    X_train,
    y_train,
    cv,
) -> pd.DataFrame:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append(
            {
                "Model": name,
                "Accuracy Mean": scores[
                    "test_accuracy"
                ].mean(),
                "Precision Mean": scores[
                    "test_precision"
                ].mean(),
                "Recall Mean": scores[
                    "test_recall"
                ].mean(),
                "F1 Mean": scores[
                    "test_f1"
                ].mean(),
                "F1 Std": scores[
                    "test_f1"
                ].std(),
                "ROC-AUC Mean": scores[
                    "test_roc_auc"
                ].mean(),
                "ROC-AUC Std": scores[
                    "test_roc_auc"
                ].std(),
            }
        )

    results_df = pd.DataFrame(
        results
    ).sort_values(
        by="F1 Mean",
        ascending=False,
    )

    return results_df


# =========================================================
# Hyperparameter Tuning
# =========================================================

def tune_logistic_regression(
    preprocessor,
    X_train,
    y_train,
    cv,
):
    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    param_grid = {
        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    }

    grid = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(
        X_train,
        y_train,
    )

    return grid


def tune_gradient_boosting(
    preprocessor,
    X_train,
    y_train,
    cv,
):
    gb_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [
            100,
            200,
        ],
        "model__learning_rate": [
            0.05,
            0.1,
        ],
        "model__max_depth": [
            2,
            3,
        ],
    }

    grid = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(
        X_train,
        y_train,
    )

    return grid


# =========================================================
# Final Model Selection
# =========================================================

def select_final_model(
    logistic_grid,
    gb_grid,
):
    if (
        logistic_grid.best_score_
        >= gb_grid.best_score_
    ):
        return (
            "Logistic Regression",
            clone(
                logistic_grid.best_estimator_
            ),
        )

    return (
        "Gradient Boosting",
        clone(
            gb_grid.best_estimator_
        ),
    )


# =========================================================
# Threshold Selection Using OOF Predictions
# =========================================================

def select_threshold(
    model,
    X_train,
    y_train,
    cv,
):
    oof_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.round(
        np.arange(
            0.20,
            0.81,
            0.01,
        ),
        2,
    )

    results = []

    for threshold in thresholds:
        predictions = (
            oof_proba >= threshold
        ).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(
                    y_train,
                    predictions,
                ),
                "Precision": precision_score(
                    y_train,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_train,
                    predictions,
                ),
                "F1": f1_score(
                    y_train,
                    predictions,
                ),
            }
        )

    threshold_results = pd.DataFrame(
        results
    )

    best_row = threshold_results.loc[
        threshold_results["F1"].idxmax()
    ]

    best_threshold = float(
        best_row["Threshold"]
    )

    return (
        best_threshold,
        best_row,
        threshold_results,
    )


# =========================================================
# Evaluation
# =========================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
        ),
        "F1": f1_score(
            y_true,
            predictions,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "PR-AUC": average_precision_score(
            y_true,
            probabilities,
        ),
    }

    return metrics


def evaluate_final_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold,
):
    train_proba = model.predict_proba(
        X_train
    )[:, 1]

    test_proba = model.predict_proba(
        X_test
    )[:, 1]

    train_metrics = calculate_metrics(
        y_train,
        train_proba,
        threshold,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_proba,
        threshold,
    )

    comparison = pd.DataFrame(
        {
            "Train": train_metrics,
            "Test": test_metrics,
        }
    )

    return (
        train_metrics,
        test_metrics,
        comparison,
    )


# =========================================================
# Model Saving
# =========================================================

def save_model(
    model,
    threshold,
):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {
        "model": model,
        "threshold": threshold,
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
    )


# =========================================================
# Main Training Workflow
# =========================================================

def main():
    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    print("\n[1/9] Loading data...")
    df = load_data()

    X, y = prepare_features(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    print(
        "Training shape:",
        X_train.shape,
    )

    print(
        "Test shape:",
        X_test.shape,
    )

    print("\n[2/9] Building preprocessing pipeline...")
    preprocessor = build_preprocessor()

    cv = build_cv()

    print("\n[3/9] Comparing candidate models...")

    comparison = compare_models(
        preprocessor,
        X_train,
        y_train,
        cv,
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\n[4/9] Tuning Logistic Regression...")

    logistic_grid = tune_logistic_regression(
        preprocessor,
        X_train,
        y_train,
        cv,
    )

    print(
        "Best parameters:",
        logistic_grid.best_params_,
    )

    print(
        "Best CV F1:",
        round(
            logistic_grid.best_score_,
            6,
        ),
    )

    print("\n[5/9] Tuning Gradient Boosting...")

    gb_grid = tune_gradient_boosting(
        preprocessor,
        X_train,
        y_train,
        cv,
    )

    print(
        "Best parameters:",
        gb_grid.best_params_,
    )

    print(
        "Best CV F1:",
        round(
            gb_grid.best_score_,
            6,
        ),
    )

    print("\n[6/9] Selecting final model...")

    (
        selected_model_name,
        selected_model,
    ) = select_final_model(
        logistic_grid,
        gb_grid,
    )

    print(
        "Selected model:",
        selected_model_name,
    )

    print(
        "\n[7/9] Selecting threshold "
        "with out-of-fold predictions..."
    )

    (
        best_threshold,
        best_threshold_row,
        _,
    ) = select_threshold(
        selected_model,
        X_train,
        y_train,
        cv,
    )

    print(
        "Selected threshold:",
        best_threshold,
    )

    print(
        "OOF F1:",
        round(
            best_threshold_row["F1"],
            6,
        ),
    )

    print("\n[8/9] Training final model...")

    final_model = clone(
        selected_model
    )

    final_model.fit(
        X_train,
        y_train,
    )

    (
        train_metrics,
        test_metrics,
        metric_comparison,
    ) = evaluate_final_model(
        final_model,
        X_train,
        y_train,
        X_test,
        y_test,
        best_threshold,
    )

    print(
        "\nTrain vs Test Metrics:"
    )

    print(
        metric_comparison.round(
            4
        ).to_string()
    )

    print(
        "\nFinal Test Metrics:"
    )

    for (
        metric_name,
        metric_value,
    ) in test_metrics.items():
        print(
            f"{metric_name}: "
            f"{metric_value:.4f}"
        )

    print("\n[9/9] Saving model...")

    save_model(
        final_model,
        best_threshold,
    )

    print(
        "Model saved to:",
        MODEL_PATH,
    )

    print(
        "Saved threshold:",
        best_threshold,
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()