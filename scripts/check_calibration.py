from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from train_baselines import FEATURES
from tune_logistic import build_pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TUNING_DIR = ROOT / "artifacts" / "tuning"
REPORT_DIR = ROOT / "reports" / "calibration"
ARTIFACT_DIR = ROOT / "artifacts" / "calibration"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(
        DATA_DIR / "game1_train.parquet"
    ).sort_values(["cutoff", "device"]).reset_index(drop=True)

    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )

    tuning_metadata = json.loads(
        (TUNING_DIR / "selected_logistic_metadata.json")
        .read_text(encoding="utf-8")
    )
    selected_c = float(tuning_metadata["selected_C"])

    # Reserve the latest approximately 20% of training times
    # for calibration. This boundary does not use target values.
    calibration_start = train.loc[
        int(len(train) * 0.80), "cutoff"
    ]

    fitting = train.loc[
        (train["cutoff"] < calibration_start)
        & (train["outcome_end"] <= calibration_start)
    ].copy()

    calibration = train.loc[
        train["cutoff"] >= calibration_start
    ].copy()

    excluded = len(train) - len(fitting) - len(calibration)

    for name, part in [
        ("model fitting", fitting),
        ("calibration", calibration),
        ("validation", validation),
    ]:
        if part.empty or part["churn"].nunique() != 2:
            raise ValueError(
                f"{name} must contain examples from both classes."
            )

    assert set(fitting["device"]).isdisjoint(calibration["device"])
    assert set(train["device"]).isdisjoint(validation["device"])
    assert fitting["outcome_end"].max() <= calibration["cutoff"].min()
    assert calibration["outcome_end"].max() <= validation["cutoff"].min()

    print(f"Selected C: {selected_c}")
    print(f"Calibration starts: {calibration_start}")
    print(f"Model-fitting devices: {len(fitting):,}")
    print(f"Calibration devices: {len(calibration):,}")
    print(f"Boundary exclusions: {excluded:,}")
    print(f"Validation devices: {len(validation):,}")
    print("Test data: NOT LOADED")

    print("\nCALIBRATION CLASS COUNTS")
    print(calibration["churn"].value_counts().sort_index().to_string())

    # Preprocessing is fitted only on the earlier fitting period.
    early_model = build_pipeline(selected_c)
    early_model.fit(
        fitting[FEATURES],
        fitting["churn"].astype(int),
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(early_model),
        method="sigmoid",
    )
    calibrated_model.fit(
        calibration[FEATURES],
        calibration["churn"].astype(int),
    )

    full_training_model = joblib.load(
        TUNING_DIR / "selected_logistic.joblib"
    )

    models = {
        "full_training_uncalibrated": full_training_model,
        "early_training_uncalibrated": early_model,
        "early_training_sigmoid": calibrated_model,
    }

    X_validation = validation[FEATURES]
    y_validation = validation["churn"].astype(int)

    results = []
    predictions = pd.DataFrame({
        "device": validation["device"],
        "actual_churn": y_validation,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    for name, model in models.items():
        churn_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            X_validation
        )[:, churn_index]

        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        results.append({
            "model": name,
            "brier_score": float(
                brier_score_loss(y_validation, probabilities)
            ),
            "log_loss": float(
                log_loss(
                    y_validation,
                    np.column_stack([1 - probabilities, probabilities]),
                    labels=[0, 1],
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_validation, probabilities)
            ),
            "return_average_precision": float(
                average_precision_score(
                    1 - y_validation, 1 - probabilities
                )
            ),
            "mean_predicted_churn": float(probabilities.mean()),
            "observed_churn_rate": float(y_validation.mean()),
        })

        predictions[name] = probabilities

        observed_rate, predicted_mean = calibration_curve(
            y_validation,
            probabilities,
            n_bins=8,
            strategy="quantile",
        )

        pd.DataFrame({
            "mean_predicted_churn": predicted_mean,
            "observed_churn_rate": observed_rate,
        }).to_csv(
            REPORT_DIR / f"{name}_curve.csv",
            index=False,
        )

        ax.plot(
            predicted_mean,
            observed_rate,
            marker="o",
            label=name,
        )

        model_path = ARTIFACT_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

        loaded = joblib.load(model_path)
        np.testing.assert_allclose(
            model.predict_proba(X_validation.iloc[:100]),
            loaded.predict_proba(X_validation.iloc[:100]),
        )

    comparison = pd.DataFrame(results).set_index("model")
    comparison.to_csv(REPORT_DIR / "comparison.csv")
    predictions.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False,
    )

    ax.set_xlabel("Mean predicted inactivity probability")
    ax.set_ylabel("Observed inactivity proportion")
    ax.set_title("Validation calibration curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "calibration_curves.png", dpi=150)
    plt.close(fig)

    metadata = {
        "C": selected_c,
        "calibration_method": "sigmoid",
        "calibration_start": str(calibration_start),
        "model_fitting_devices": len(fitting),
        "calibration_devices": len(calibration),
        "boundary_exclusions": excluded,
        "validation_devices": len(validation),
        "features": FEATURES,
        "test_loaded": False,
    }
    (ARTIFACT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nVALIDATION CALIBRATION COMPARISON")
    print(comparison.round(6).to_string())

    print(f"\nReports saved to: {REPORT_DIR}")
    print("Saved-model prediction checks passed.")
    print("No model was fitted on validation data.")
    print("Do not refit the base model beneath a fitted calibrator.")


if __name__ == "__main__":
    main()