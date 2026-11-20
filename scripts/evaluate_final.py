from pathlib import Path
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from analyze_decisions import ranking_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
POLICY_PATH = ROOT / "reports" / "decisions" / "decision_policy.json"

RELEASE_DIR = ROOT / "artifacts" / "releases" / "v1"
REPORT_DIR = ROOT / "reports" / "final_test"
RESULT_PATH = REPORT_DIR / "metrics.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_metrics(y_true, probabilities):
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "churn_average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "return_average_precision": float(
            average_precision_score(1 - y_true, 1 - probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack([1 - probabilities, probabilities]),
                labels=[0, 1],
            )
        ),
    }


def main():
    # Protect a completed evaluation against accidental replacement.
    if RESULT_PATH.exists():
        raise RuntimeError(
            "Final test results already exist. "
            "Review the saved results instead of overwriting them."
        )

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    source_model = ROOT / policy["model_path"]

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    frozen_model_path = RELEASE_DIR / "model.joblib"
    frozen_policy_path = RELEASE_DIR / "decision_policy.json"

    # Allow recovery from an interrupted run only if inputs are unchanged.
    for source, destination in [
        (source_model, frozen_model_path),
        (POLICY_PATH, frozen_policy_path),
    ]:
        if destination.exists():
            if file_hash(source) != file_hash(destination):
                raise RuntimeError(
                    f"Frozen release differs from current input: {destination}"
                )
        else:
            shutil.copy2(source, destination)

    model = joblib.load(frozen_model_path)
    features = policy["features"]
    threshold = float(policy["threshold"])

    print("Model and decision policy frozen as v1.")
    print(f"Fixed threshold: {threshold:.2f}")
    print("Loading the reserved test partition now.")

    train = pd.read_parquet(DATA_DIR / "game1_train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )
    test = pd.read_parquet(DATA_DIR / "game1_test.parquet")

    assert set(train["device"]).isdisjoint(test["device"])
    assert set(validation["device"]).isdisjoint(test["device"])
    assert validation["outcome_end"].max() <= test["cutoff"].min()
    assert test["churn"].nunique() == 2

    y_test = test["churn"].astype(int).to_numpy()
    X_test = test[features]

    churn_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, churn_index]

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    predictions = (probabilities >= threshold).astype(int)

    metrics = probability_metrics(y_test, probabilities)
    metrics["accuracy"] = float(
        accuracy_score(y_test, predictions)
    )
    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_test, predictions)
    )

    class_report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["returned", "inactive"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    # Baseline probabilities come from TRAINING prevalence only.
    training_prior = float(train["churn"].mean())
    dummy_probabilities = np.full(len(test), training_prior)
    dummy_predictions = (
        dummy_probabilities >= 0.5
    ).astype(int)

    dummy_metrics = probability_metrics(
        y_test, dummy_probabilities
    )
    dummy_metrics["accuracy"] = float(
        accuracy_score(y_test, dummy_predictions)
    )
    dummy_metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_test, dummy_predictions)
    )

    recency_scores = test["days_since_last_play"].to_numpy()
    recency_metrics = {
        "roc_auc": float(
            roc_auc_score(y_test, recency_scores)
        ),
        "churn_average_precision": float(
            average_precision_score(y_test, recency_scores)
        ),
        "return_average_precision": float(
            average_precision_score(1 - y_test, -recency_scores)
        ),
    }

    ranking = pd.DataFrame([
        ranking_metrics(
            "logistic_regression",
            y_test,
            probabilities,
            fraction=policy["ranking_capacity_fraction"],
        ),
        ranking_metrics(
            "recency",
            y_test,
            recency_scores,
            fraction=policy["ranking_capacity_fraction"],
        ),
    ])

    pd.DataFrame({
        "device": test["device"],
        "actual_churn": y_test,
        "churn_probability": probabilities,
        "predicted_churn": predictions,
        "recency_score": recency_scores,
    }).to_csv(
        REPORT_DIR / "predictions.csv",
        index=False,
    )

    ranking.to_csv(
        REPORT_DIR / "top_10_percent_comparison.csv",
        index=False,
    )

    result = {
        "release": "v1",
        "threshold": threshold,
        "test_devices": len(test),
        "test_churn_prevalence": float(y_test.mean()),
        "model_sha256": file_hash(frozen_model_path),
        "policy_sha256": file_hash(frozen_policy_path),
        "test_file_sha256": file_hash(
            DATA_DIR / "game1_test.parquet"
        ),
        "model_metrics": metrics,
        "classification_report": class_report,
        "confusion_matrix": matrix.tolist(),
        "dummy_metrics": dummy_metrics,
        "recency_ranking_metrics": recency_metrics,
        "top_risk_ranking": ranking.to_dict(orient="records"),
        "limitations": [
            "Earlier descriptive EDA included the full dataset.",
            "Model and threshold selection used validation data, not test.",
            "Devices are proxies for players.",
            "Labels represent windowed inactivity, not permanent churn.",
            "Retention intervention effectiveness has not been tested.",
        ],
    }

    RESULT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\nTEST DEVICES: {len(test):,}")
    print(f"OBSERVED INACTIVITY: {y_test.mean():.2%}")

    print("\nFINAL MODEL CLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["returned", "inactive"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix: rows=actual, columns=predicted")
    print("Class order: returned (0), inactive (1)")
    print(matrix)

    print("\nMODEL VS TRAINING-PRIOR BASELINE")
    print(pd.DataFrame({
        "selected_model": metrics,
        "dummy": dummy_metrics,
    }).round(6).to_string())

    print("\nRECENCY RANKING METRICS")
    for name, value in recency_metrics.items():
        print(f"{name}: {value:.6f}")

    print("\nTOP 10% RISK LIST")
    print(ranking.round(4).to_string(index=False))

    print(f"\nFrozen release: {RELEASE_DIR}")
    print(f"Final results: {RESULT_PATH}")
    print("Do not tune the model or threshold using these test results.")


if __name__ == "__main__":
    main()