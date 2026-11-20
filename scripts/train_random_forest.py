from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Reuse the same feature list and evaluation function.
# This import works when running this script from the scripts folder path.
from train_baselines import FEATURES, evaluate_probabilities


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "baselines"
ARTIFACT_DIR = ROOT / "artifacts" / "baselines"


def build_random_forest(class_weight=None):
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
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight=class_weight,
                random_state=42,
                n_jobs=2,
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

    print(f"Training examples: {len(train):,}")
    print(f"Validation examples: {len(validation):,}")
    print("Test data: NOT LOADED")
    print("Training 300-tree forests with at most 2 CPU workers.")

    experiments = {
        "random_forest": None,
        "weighted_random_forest": "balanced",
    }

    results = []
    predictions = pd.DataFrame({
        "device": validation["device"],
        "actual_churn": y_validation,
    })

    for name, class_weight in experiments.items():
        print(f"\nTraining: {name}")

        pipeline = build_random_forest(
            class_weight=class_weight
        )
        pipeline.fit(X_train, y_train)

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

        # Confirm that saving/loading preserves predictions.
        loaded_pipeline = joblib.load(model_path)
        sample = X_validation.iloc[:100]

        np.testing.assert_allclose(
            pipeline.predict_proba(sample),
            loaded_pipeline.predict_proba(sample),
        )

        metadata = {
            "model": name,
            "features": FEATURES,
            "training_examples": len(train),
            "validation_examples": len(validation),
            "evaluation_partition": "validation",
            "threshold": 0.5,
            "parameters": pipeline.named_steps[
                "model"
            ].get_params(),
        }

        (ARTIFACT_DIR / f"{name}_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    forest_results = pd.DataFrame(results).set_index("model")

    forest_results.to_csv(
        REPORT_DIR / "random_forest_comparison.csv"
    )
    predictions.to_csv(
        REPORT_DIR / "random_forest_validation_predictions.csv",
        index=False,
    )

    # Combine with the earlier baseline results without replacing them.
    baseline_path = REPORT_DIR / "model_comparison.csv"

    if baseline_path.exists():
        baselines = pd.read_csv(
            baseline_path,
            index_col="model",
        )
        comparison = pd.concat([baselines, forest_results])
    else:
        comparison = forest_results

    comparison.to_csv(
        REPORT_DIR / "comparison_through_random_forest.csv"
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


if __name__ == "__main__":
    main()