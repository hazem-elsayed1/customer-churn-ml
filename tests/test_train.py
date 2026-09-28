import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from src.train import (
    calculate_metrics,
    select_final_model,
    select_threshold,
)


def test_calculate_metrics_returns_expected_values():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = calculate_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.5,
    )

    assert set(metrics.keys()) == {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    }

    assert metrics["Accuracy"] == 0.75
    assert metrics["Precision"] == 2 / 3
    assert metrics["Recall"] == 1.0
    assert metrics["F1"] == 0.8


class FakeGridSearch:
    def __init__(self, best_score, best_estimator):
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator


def test_select_final_model_chooses_logistic_regression():
    logistic_model = DummyClassifier(strategy="most_frequent")
    gb_model = DummyClassifier(strategy="prior")

    logistic_grid = FakeGridSearch(
        best_score=0.65,
        best_estimator=logistic_model,
    )

    gb_grid = FakeGridSearch(
        best_score=0.60,
        best_estimator=gb_model,
    )

    selected_name, selected_model = select_final_model(
        logistic_grid,
        gb_grid,
    )

    assert selected_name == "Logistic Regression"
    assert isinstance(selected_model, DummyClassifier)
    assert selected_model.strategy == "most_frequent"


def test_select_final_model_chooses_gradient_boosting():
    logistic_model = DummyClassifier(strategy="most_frequent")
    gb_model = DummyClassifier(strategy="prior")

    logistic_grid = FakeGridSearch(
        best_score=0.55,
        best_estimator=logistic_model,
    )

    gb_grid = FakeGridSearch(
        best_score=0.70,
        best_estimator=gb_model,
    )

    selected_name, selected_model = select_final_model(
        logistic_grid,
        gb_grid,
    )

    assert selected_name == "Gradient Boosting"
    assert isinstance(selected_model, DummyClassifier)
    assert selected_model.strategy == "prior"


def test_select_threshold_returns_valid_threshold_and_results():
    X_train = pd.DataFrame(
        {
            "feature": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )

    y_train = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    )

    model = DummyClassifier(strategy="prior")

    cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    threshold, best_row, threshold_results = select_threshold(
        model=model,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
    )

    assert 0.20 <= threshold <= 0.81

    assert isinstance(best_row, pd.Series)

    assert "Threshold" in best_row.index
    assert "Accuracy" in best_row.index
    assert "Precision" in best_row.index
    assert "Recall" in best_row.index
    assert "F1" in best_row.index

    assert isinstance(threshold_results, pd.DataFrame)

    assert "Threshold" in threshold_results.columns
    assert "Accuracy" in threshold_results.columns
    assert "Precision" in threshold_results.columns
    assert "Recall" in threshold_results.columns
    assert "F1" in threshold_results.columns

    assert len(threshold_results) > 0
