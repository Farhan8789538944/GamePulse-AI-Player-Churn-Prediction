from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "baselines"
ARTIFACT_DIR = ROOT / "artifacts" / "baselines"

FEATURES = [
    "plays_7d",
    "active_days_7d",
    "days_since_last_play",
    "activity_span_days",
    "plays_first_3d",
    "plays_last_3d",
    "play_count_change",
    "mean_score_7d",
    "best_score_7d",
    "score_std_7d",
    "mean_gap_hours",
    "max_gap_hours",
]


def build_logistic_pipeline(class_weight=None):
    """Create an unweighted or class-weighted model pipeline."""
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight=class_weight,
                random_state=42,
            ),
        ),
    ])


def churn_probabilities(model, features):
    """Return probabilities for class 1: future inactivity."""
    churn_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, churn_index]


def evaluate_probabilities(name, y_true, probabilities):
    """Evaluate validation predictions at threshold 0.5."""
    probabilities = np.asarray(probabilities)

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: non-finite probabilities.")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(f"{name}: probabilities outside [0, 1].")

    predictions = (probabilities >= 0.5).astype(int)

    class_metrics = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["returned", "inactive"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    metrics = {
        "model": name,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "churn_average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "return_average_precision": float(
            average_precision_score(
                1 - y_true, 1 - probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "churn_precision": class_metrics["inactive"]["precision"],
        "churn_recall": class_metrics["inactive"]["recall"],
        "churn_f1": class_metrics["inactive"]["f1-score"],
        "return_precision": class_metrics["returned"]["precision"],
        "return_recall": class_metrics["returned"]["recall"],
        "return_f1": class_metrics["returned"]["f1-score"],
    }

    print(f"\n{name.upper()}")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["returned", "inactive"],
        digits=3,
        zero_division=0,
    ))

    print("Confusion matrix: rows=actual, columns=predicted")
    print("Class order: returned (0), inactive (1)")
    print(matrix)

    details = {
        "metrics": metrics,
        "classification_report": class_metrics,
        "confusion_matrix": matrix.tolist(),
        "threshold": 0.5,
        "evaluation_partition": "validation",
    }

    (REPORT_DIR / f"{name}.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    return metrics


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the test partition untouched.
    train = pd.read_parquet(
        DATA_DIR / "game1_train.parquet"
    )
    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )

    X_train = train[FEATURES]
    y_train = train["churn"].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation["churn"].astype(int)

    assert set(train["device"]).isdisjoint(validation["device"])
    assert train["outcome_end"].max() <= validation["cutoff"].min()
    assert y_train.nunique() == 2
    assert y_validation.nunique() == 2

    print(f"Training examples: {len(train):,}")
    print(f"Validation examples: {len(validation):,}")
    print("Test data: NOT LOADED")

    # Show the weights calculated from training data only.
    class_counts = y_train.value_counts().sort_index()
    class_weights = {
        int(label): len(y_train) / (2 * int(count))
        for label, count in class_counts.items()
    }

    print("\nBALANCED TRAINING WEIGHTS")
    for label, weight in class_weights.items():
        print(f"Class {label}: {weight:.4f}")

    # Baseline 1: class-prior probabilities.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(train), 1)), y_train)

    dummy_probabilities = churn_probabilities(
        dummy,
        np.zeros((len(validation), 1)),
    )

    # Baseline 2: standard Logistic Regression.
    logistic = build_logistic_pipeline()
    logistic.fit(X_train, y_train)

    logistic_probabilities = churn_probabilities(
        logistic, X_validation
    )

    # Module 6, Step 1: weighted Logistic Regression.
    weighted_logistic = build_logistic_pipeline(
        class_weight="balanced"
    )
    weighted_logistic.fit(X_train, y_train)

    weighted_probabilities = churn_probabilities(
        weighted_logistic, X_validation
    )

    rows = [
        evaluate_probabilities(
            "dummy",
            y_validation,
            dummy_probabilities,
        ),
        evaluate_probabilities(
            "logistic_regression",
            y_validation,
            logistic_probabilities,
        ),
        evaluate_probabilities(
            "weighted_logistic_regression",
            y_validation,
            weighted_probabilities,
        ),
    ]

    comparison = pd.DataFrame(rows).set_index("model")
    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv"
    )

    # Recency is a ranking score, not a calibrated probability.
    recency_scores = validation[
        "days_since_last_play"
    ].to_numpy()

    recency_metrics = {
        "roc_auc": float(
            roc_auc_score(y_validation, recency_scores)
        ),
        "churn_average_precision": float(
            average_precision_score(
                y_validation, recency_scores
            )
        ),
        "return_average_precision": float(
            average_precision_score(
                1 - y_validation, -recency_scores
            )
        ),
    }

    (REPORT_DIR / "recency_ranking.json").write_text(
        json.dumps(recency_metrics, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        logistic,
        ARTIFACT_DIR / "logistic_regression.joblib",
    )
    joblib.dump(
        weighted_logistic,
        ARTIFACT_DIR / "weighted_logistic_regression.joblib",
    )

    (ARTIFACT_DIR / "feature_columns.json").write_text(
        json.dumps(FEATURES, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({
        "device": validation["device"],
        "actual_churn": y_validation,
        "dummy_probability": dummy_probabilities,
        "logistic_probability": logistic_probabilities,
        "weighted_logistic_probability": weighted_probabilities,
        "recency_score": recency_scores,
    }).to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False,
    )

    print("\nVALIDATION MODEL COMPARISON")
    print(comparison.round(4).to_string())

    print("\nRECENCY RANKING BASELINE")
    for metric, value in recency_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nReports saved to: {REPORT_DIR}")
    print(f"Models saved to: {ARTIFACT_DIR}")
    print("These are experiments, not the selected final model.")
    print("Class-weighted probabilities may require calibration.")


if __name__ == "__main__":
    main()