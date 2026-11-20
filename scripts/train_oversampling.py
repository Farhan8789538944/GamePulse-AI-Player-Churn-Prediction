from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_baselines import FEATURES, evaluate_probabilities


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "baselines"
ARTIFACT_DIR = ROOT / "artifacts" / "baselines"


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
    y_train = train["churn"].astype(int).to_numpy()

    X_validation = validation[FEATURES]
    y_validation = validation["churn"].astype(int)

    assert set(train["device"]).isdisjoint(validation["device"])
    assert train["outcome_end"].max() <= validation["cutoff"].min()
    assert set(np.unique(y_train)) == {0, 1}
    assert y_validation.nunique() == 2

    print(f"Training examples before sampling: {len(train):,}")
    print(f"Validation examples: {len(validation):,}")
    print("Test data: NOT LOADED")

    # Fit preprocessing on ORIGINAL training data only.
    preprocessing = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    X_train_transformed = preprocessing.fit_transform(X_train)

    # Random oversampling: keep every original row and duplicate
    # minority-class rows until both classes have equal counts.
    rng = np.random.default_rng(42)
    counts_before = pd.Series(y_train).value_counts().sort_index()
    target_count = int(counts_before.max())

    index_parts = []

    for label in [0, 1]:
        original_indices = np.flatnonzero(y_train == label)
        index_parts.append(original_indices)

        extra_count = target_count - len(original_indices)

        if extra_count > 0:
            duplicate_indices = rng.choice(
                original_indices,
                size=extra_count,
                replace=True,
            )
            index_parts.append(duplicate_indices)

    sampled_indices = np.concatenate(index_parts)
    rng.shuffle(sampled_indices)

    X_resampled = X_train_transformed[sampled_indices]
    y_resampled = y_train[sampled_indices]

    counts_after = (
        pd.Series(y_resampled).value_counts().sort_index()
    )

    assert counts_after.loc[0] == counts_after.loc[1]
    assert np.unique(sampled_indices).size == len(train)

    print("\nCLASS COUNTS BEFORE SAMPLING")
    print(counts_before.to_string())

    print("\nCLASS COUNTS AFTER SAMPLING")
    print(counts_after.to_string())

    print(f"\nResampled training examples: {len(y_resampled):,}")

    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight=None,
        random_state=42,
    )

    model.fit(X_resampled, y_resampled)

    # Assemble the already-fitted components for inference.
    # Do not call fit() on this assembled pipeline: doing so would
    # bypass the explicit resampling procedure above.
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])

    churn_index = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(
        X_validation
    )[:, churn_index]

    name = "oversampled_logistic_regression"

    result = evaluate_probabilities(
        name,
        y_validation,
        probabilities,
    )

    model_path = ARTIFACT_DIR / f"{name}.joblib"
    joblib.dump(pipeline, model_path)

    loaded = joblib.load(model_path)

    np.testing.assert_allclose(
        pipeline.predict_proba(X_validation.iloc[:100]),
        loaded.predict_proba(X_validation.iloc[:100]),
    )

    experiment_results = (
        pd.DataFrame([result]).set_index("model")
    )

    experiment_results.to_csv(
        REPORT_DIR / "oversampling_comparison.csv"
    )

    pd.DataFrame({
        "device": validation["device"],
        "actual_churn": y_validation,
        "oversampled_logistic_probability": probabilities,
    }).to_csv(
        REPORT_DIR / "oversampling_validation_predictions.csv",
        index=False,
    )

    metadata = {
        "model": name,
        "features": FEATURES,
        "sampling": "random oversampling to equal class counts",
        "sampling_seed": 42,
        "preprocessing_fit_on": "original training data",
        "original_training_examples": len(train),
        "resampled_training_examples": len(y_resampled),
        "validation_examples": len(validation),
        "class_counts_before": {
            str(label): int(count)
            for label, count in counts_before.items()
        },
        "class_counts_after": {
            str(label): int(count)
            for label, count in counts_after.items()
        },
        "threshold": 0.5,
        "evaluation_partition": "validation",
        "parameters": model.get_params(),
    }

    (ARTIFACT_DIR / f"{name}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    comparison_parts = []

    for filename in [
        "model_comparison.csv",
        "random_forest_comparison.csv",
        "xgboost_comparison.csv",
    ]:
        path = REPORT_DIR / filename

        if path.exists():
            comparison_parts.append(
                pd.read_csv(path, index_col="model")
            )
        else:
            print(f"Previous comparison not found: {filename}")

    comparison_parts.append(experiment_results)
    comparison = pd.concat(comparison_parts)

    comparison.to_csv(
        REPORT_DIR / "comparison_through_oversampling.csv"
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
    print(f"Model saved to: {model_path}")
    print("Saved-model prediction check passed.")
    print("Validation data was not resampled.")
    print("Oversampling does not create new independent observations.")
    print("Resampled-model probabilities may require calibration.")


if __name__ == "__main__":
    main()