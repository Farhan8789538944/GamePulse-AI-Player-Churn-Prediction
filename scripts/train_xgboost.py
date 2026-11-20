from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train_baselines import FEATURES, evaluate_probabilities


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "baselines"
ARTIFACT_DIR = ROOT / "artifacts" / "baselines"


def build_xgboost():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "model",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                tree_method="hist",
                device="cpu",
                n_jobs=2,
                random_state=42,
            ),
        ),
    ])


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"XGBoost version: {xgboost.__version__}")
    print(f"Training examples: {len(train):,}")
    print(f"Validation examples: {len(validation):,}")
    print("Test data: NOT LOADED")
    print("Training on CPU with at most 2 workers.")

    # Give the two classes equal total weight in the weighted experiment.
    # This correctly upweights class 0, our minority class.
    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    experiments = {
        "xgboost": None,
        "weighted_xgboost": balanced_weights,
    }

    results = []
    predictions = pd.DataFrame({
        "device": validation["device"],
        "actual_churn": y_validation,
    })

    for name, weights in experiments.items():
        print(f"\nTraining: {name}")

        pipeline = build_xgboost()

        if weights is None:
            pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(
                X_train,
                y_train,
                model__sample_weight=weights,
            )

        churn_index = list(pipeline.classes_).index(1)

        probabilities = pipeline.predict_proba(
            X_validation
        )[:, churn_index]

        results.append(
            evaluate_probabilities(
                name,
                y_validation,
                probabilities,
            )
        )

        predictions[f"{name}_probability"] = probabilities

        model_path = ARTIFACT_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

        # Verify model serialization.
        loaded_pipeline = joblib.load(model_path)
        sample = X_validation.iloc[:100]

        np.testing.assert_allclose(
            pipeline.predict_proba(sample),
            loaded_pipeline.predict_proba(sample),
        )

        metadata = {
            "model": name,
            "xgboost_version": xgboost.__version__,
            "features": FEATURES,
            "training_examples": len(train),
            "validation_examples": len(validation),
            "evaluation_partition": "validation",
            "threshold": 0.5,
            "weighting": (
                "balanced sample weights"
                if weights is not None
                else "none"
            ),
            "parameters": pipeline.named_steps[
                "model"
            ].get_params(),
        }

        (ARTIFACT_DIR / f"{name}_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    xgb_results = pd.DataFrame(results).set_index("model")
    xgb_results.to_csv(
        REPORT_DIR / "xgboost_comparison.csv"
    )

    predictions.to_csv(
        REPORT_DIR / "xgboost_validation_predictions.csv",
        index=False,
    )

    # Collect previous results without overwriting their files.
    comparison_parts = []

    for filename in [
        "model_comparison.csv",
        "random_forest_comparison.csv",
    ]:
        path = REPORT_DIR / filename

        if path.exists():
            comparison_parts.append(
                pd.read_csv(path, index_col="model")
            )
        else:
            print(f"Previous comparison not found: {filename}")

    comparison_parts.append(xgb_results)
    comparison = pd.concat(comparison_parts)

    comparison.to_csv(
        REPORT_DIR / "comparison_through_xgboost.csv"
    )

    display_columns = [
        "balanced_accuracy",
        "roc_auc",
        "return_average_precision",
        "churn_average_precision",
        "brier_score",
        "return_precision",
        "return_recall",
    ]

    print("\nVALIDATION COMPARISON")
    print(
        comparison[display_columns]
        .round(4)
        .to_string()
    )

    print(f"\nReports saved to: {REPORT_DIR}")
    print(f"Models saved to: {ARTIFACT_DIR}")
    print("Saved-model prediction checks passed.")
    print("No final model has been selected.")
    print("Weighted probabilities may require calibration.")


if __name__ == "__main__":
    main()