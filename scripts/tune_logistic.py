from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baselines import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "tuning"
ARTIFACT_DIR = ROOT / "artifacts" / "tuning"

C_VALUES = [0.01, 0.1, 1.0, 10.0]


def build_pipeline(c_value):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=c_value,
                class_weight=None,
                max_iter=3000,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_DIR / "game1_train.parquet")
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
    print("Selection criterion: lowest validation Brier score")

    results = []
    candidates = {}
    candidate_probabilities = {}

    for c_value in C_VALUES:
        print(f"\nTraining Logistic Regression with C={c_value}")

        pipeline = build_pipeline(c_value)
        pipeline.fit(X_train, y_train)

        churn_index = list(pipeline.classes_).index(1)
        probabilities = pipeline.predict_proba(
            X_validation
        )[:, churn_index]

        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        results.append({
            "C": c_value,
            "brier_score": float(
                brier_score_loss(y_validation, probabilities)
            ),
            "log_loss": float(
                log_loss(
                    y_validation,
                    np.column_stack([
                        1 - probabilities,
                        probabilities,
                    ]),
                    labels=[0, 1],
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_validation, probabilities)
            ),
            "return_average_precision": float(
                average_precision_score(
                    1 - y_validation,
                    1 - probabilities,
                )
            ),
            "churn_average_precision": float(
                average_precision_score(
                    y_validation,
                    probabilities,
                )
            ),
        })

        candidates[c_value] = pipeline
        candidate_probabilities[c_value] = probabilities

    # In an exact tie, prefer stronger regularisation (smaller C).
    comparison = pd.DataFrame(results).sort_values(
        ["brier_score", "C"]
    )

    comparison.to_csv(
        REPORT_DIR / "logistic_tuning.csv",
        index=False,
    )

    best_c = float(comparison.iloc[0]["C"])
    selected_pipeline = candidates[best_c]
    selected_probabilities = candidate_probabilities[best_c]

    model_path = ARTIFACT_DIR / "selected_logistic.joblib"
    joblib.dump(selected_pipeline, model_path)

    loaded_pipeline = joblib.load(model_path)

    np.testing.assert_allclose(
        selected_pipeline.predict_proba(X_validation.iloc[:100]),
        loaded_pipeline.predict_proba(X_validation.iloc[:100]),
    )

    pd.DataFrame({
        "device": validation["device"],
        "actual_churn": y_validation,
        "churn_probability": selected_probabilities,
    }).to_csv(
        REPORT_DIR / "selected_logistic_validation_predictions.csv",
        index=False,
    )

    metadata = {
        "model": "unweighted_logistic_regression",
        "selected_C": best_c,
        "selection_metric": "validation_brier_score",
        "features": FEATURES,
        "training_examples": len(train),
        "validation_examples": len(validation),
        "candidate_C_values": C_VALUES,
        "parameters": selected_pipeline.named_steps[
            "model"
        ].get_params(),
        "status": "candidate_pending_calibration_and_final_evaluation",
    }

    (ARTIFACT_DIR / "selected_logistic_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nVALIDATION TUNING RESULTS")
    print(comparison.round(6).to_string(index=False))

    print(f"\nSelected C: {best_c}")
    print(f"Candidate saved to: {model_path}")
    print("Saved-model prediction check passed.")
    print("No threshold has been tuned.")
    print("The test partition remains reserved.")


if __name__ == "__main__":
    main()