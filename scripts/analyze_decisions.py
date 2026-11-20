from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "decisions"
MODEL_PATH = (
    ROOT / "artifacts" / "tuning" / "selected_logistic.joblib"
)

# Predefined threshold grid, including the default 0.5.
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.01), 2)


def threshold_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "return_precision": float(precision[0]),
        "return_recall": float(recall[0]),
        "return_f1": float(f1[0]),
        "churn_precision": float(precision[1]),
        "churn_recall": float(recall[1]),
        "churn_f1": float(f1[1]),
        "returned_predicted_returned": int(matrix[0, 0]),
        "returned_predicted_inactive": int(matrix[0, 1]),
        "inactive_predicted_returned": int(matrix[1, 0]),
        "inactive_predicted_inactive": int(matrix[1, 1]),
    }


def ranking_metrics(name, y_true, scores, fraction=0.10):
    """
    Evaluate a fixed-capacity high-risk list.

    When scores tie at the selection boundary, allocate the remaining
    positions proportionally across the tied group. This represents
    expected performance under random tie-breaking.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    k = max(1, int(np.ceil(len(y_true) * fraction)))
    boundary = np.sort(scores)[-k]

    above = scores > boundary
    tied = scores == boundary

    remaining_slots = k - int(above.sum())
    tie_fraction = remaining_slots / int(tied.sum())

    expected_churners = (
        y_true[above].sum()
        + tie_fraction * y_true[tied].sum()
    )

    prevalence = float(y_true.mean())
    precision_at_k = float(expected_churners / k)
    recall_at_k = float(expected_churners / y_true.sum())

    return {
        "ranking": name,
        "selected_devices": k,
        "expected_churners_selected": float(expected_churners),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "lift_over_random": precision_at_k / prevalence,
        "boundary_tie_size": int(tied.sum()),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Only validation is needed; test remains unopened.
    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )

    metadata = json.loads(
        (
            ROOT / "artifacts" / "tuning"
            / "selected_logistic_metadata.json"
        ).read_text(encoding="utf-8")
    )
    features = metadata["features"]

    model = joblib.load(MODEL_PATH)

    X_validation = validation[features]
    y_validation = validation["churn"].astype(int).to_numpy()

    churn_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        X_validation
    )[:, churn_index]

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert set(np.unique(y_validation)) == {0, 1}

    print(f"Validation devices: {len(validation):,}")
    print("Test data: NOT LOADED")
    print("Model: uncalibrated Logistic Regression, C=0.01")

    results = pd.DataFrame([
        threshold_metrics(y_validation, probabilities, threshold)
        for threshold in THRESHOLDS
    ])

    # Deterministic tie-break:
    # prefer the threshold closest to 0.5, then the lower threshold.
    results["distance_from_default"] = (
        results["threshold"] - 0.5
    ).abs()

    ordered = results.sort_values(
        ["balanced_accuracy", "distance_from_default", "threshold"],
        ascending=[False, True, True],
    )

    selected_threshold = float(ordered.iloc[0]["threshold"])

    results.drop(columns="distance_from_default").to_csv(
        REPORT_DIR / "threshold_comparison.csv",
        index=False,
    )

    columns = [
        "threshold",
        "balanced_accuracy",
        "return_precision",
        "return_recall",
        "churn_precision",
        "churn_recall",
    ]

    print("\nTOP FIVE VALIDATION THRESHOLDS")
    print(ordered[columns].head(5).round(4).to_string(index=False))

    default_metrics = threshold_metrics(
        y_validation, probabilities, 0.5
    )
    selected_metrics = threshold_metrics(
        y_validation, probabilities, selected_threshold
    )

    print("\nDEFAULT VS SELECTED THRESHOLD")
    print(
        pd.DataFrame([default_metrics, selected_metrics])[columns]
        .round(4)
        .to_string(index=False)
    )

    selected_predictions = (
        probabilities >= selected_threshold
    ).astype(int)

    print("\nSELECTED-THRESHOLD CONFUSION MATRIX")
    print("Rows=actual, columns=predicted; order=[returned, inactive]")
    print(confusion_matrix(
        y_validation,
        selected_predictions,
        labels=[0, 1],
    ))

    ranking_rows = [
        ranking_metrics(
            "logistic_regression",
            y_validation,
            probabilities,
        ),
        ranking_metrics(
            "recency",
            y_validation,
            validation["days_since_last_play"].to_numpy(),
        ),
    ]

    ranking_table = pd.DataFrame(ranking_rows)
    ranking_table.to_csv(
        REPORT_DIR / "top_10_percent_comparison.csv",
        index=False,
    )

    prevalence = float(y_validation.mean())
    k = int(ranking_rows[0]["selected_devices"])

    print("\nTOP 10% HIGH-RISK LIST")
    print(ranking_table.round(4).to_string(index=False))

    print("\nRANDOM-SELECTION EXPECTATION")
    print(f"Selected devices: {k}")
    print(f"Expected precision: {prevalence:.4f}")
    print(f"Expected recall: {k / len(validation):.4f}")
    print("Expected lift: 1.0000")
    print(f"Maximum possible precision lift: {1 / prevalence:.4f}")

    policy = {
        "model_path": "artifacts/tuning/selected_logistic.joblib",
        "model_type": "unweighted_logistic_regression",
        "C": float(metadata["selected_C"]),
        "calibration": "none",
        "features": features,
        "positive_label": 1,
        "positive_label_meaning": "no gameplay in the outcome window",
        "threshold": selected_threshold,
        "threshold_selection_metric": "validation_balanced_accuracy",
        "threshold_candidates": THRESHOLDS.tolist(),
        "ranking_capacity_fraction": 0.10,
        "status": "development_policy_pending_final_test",
    }

    (REPORT_DIR / "decision_policy.json").write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8",
    )

    print(f"\nSelected threshold: {selected_threshold:.2f}")
    print(f"Reports saved to: {REPORT_DIR}")
    print("Validation threshold performance is selection-biased.")
    print("Final test evaluation has not been performed.")


if __name__ == "__main__":
    main()