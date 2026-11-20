from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap

from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RELEASE_DIR = ROOT / "artifacts" / "releases" / "v1"
REPORT_DIR = ROOT / "reports" / "explanations"
EXPLAINER_DIR = ROOT / "artifacts" / "explanations" / "v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINER_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RELEASE_DIR / "model.joblib"
    policy_path = RELEASE_DIR / "decision_policy.json"

    final_results = json.loads(
        (ROOT / "reports" / "final_test" / "metrics.json")
        .read_text(encoding="utf-8")
    )

    if sha256(model_path) != final_results["model_sha256"]:
        raise ValueError("Frozen model hash does not match final evaluation.")

    if sha256(policy_path) != final_results["policy_sha256"]:
        raise ValueError("Frozen policy hash does not match final evaluation.")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    features = policy["features"]
    threshold = float(policy["threshold"])

    pipeline = joblib.load(model_path)

    train = pd.read_parquet(DATA_DIR / "game1_train.parquet")
    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )

    # Fixed, reproducible samples; not selected by model outcome.
    background = train.sample(
        n=min(200, len(train)),
        random_state=42,
    )
    sample = validation.sample(
        n=min(500, len(validation)),
        random_state=42,
    ).reset_index(drop=True)

    # Apply the already-fitted preprocessing. Never fit again here.
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["model"]

    background_transformed = scaler.transform(
        imputer.transform(background[features])
    )
    sample_transformed = scaler.transform(
        imputer.transform(sample[features])
    )

    transformed_names = imputer.get_feature_names_out(features)

    masker = shap.maskers.Independent(
        background_transformed,
        max_samples=len(background_transformed),
    )

    explainer = shap.LinearExplainer(
        classifier,
        masker,
        feature_names=transformed_names.tolist(),
    )

    explanations = explainer(sample_transformed)

    # SHAP values must reconstruct the model's log-odds.
    reconstructed_logits = (
        np.asarray(explanations.base_values).reshape(-1)
        + explanations.values.sum(axis=1)
    )

    model_logits = pipeline.decision_function(sample[features])

    np.testing.assert_allclose(
        reconstructed_logits,
        model_logits,
        rtol=1e-6,
        atol=1e-6,
    )

    churn_index = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(
        sample[features]
    )[:, churn_index]

    np.testing.assert_allclose(
        expit(reconstructed_logits),
        probabilities,
        rtol=1e-6,
        atol=1e-6,
    )

    importance = pd.DataFrame({
        "feature": transformed_names,
        "mean_absolute_shap_log_odds": np.abs(
            explanations.values
        ).mean(axis=0),
    }).sort_values(
        "mean_absolute_shap_log_odds",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR / "global_importance.csv",
        index=False,
    )

    shap.plots.bar(explanations, max_display=14, show=False)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "global_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    shap.plots.beeswarm(
        explanations,
        max_display=14,
        show=False,
    )
    plt.savefig(
        REPORT_DIR / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    # Explain the first randomly sampled device, without cherry-picking.
    row_index = 0
    shap.plots.waterfall(
        explanations[row_index],
        max_display=14,
        show=False,
    )
    plt.savefig(
        REPORT_DIR / "example_device_waterfall.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    contributions = pd.DataFrame({
        "feature": transformed_names,
        "shap_log_odds": explanations.values[row_index],
        "transformed_value": sample_transformed[row_index],
    })

    contributions["direction"] = np.where(
        contributions["shap_log_odds"] > 0,
        "increases predicted inactivity",
        np.where(
            contributions["shap_log_odds"] < 0,
            "decreases predicted inactivity",
            "no contribution",
        ),
    )

    contributions["absolute_contribution"] = (
        contributions["shap_log_odds"].abs()
    )
    contributions = contributions.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    contributions.to_csv(
        REPORT_DIR / "example_device_contributions.csv",
        index=False,
    )

    pd.DataFrame({
        "device": sample["device"],
        "churn_probability": probabilities,
        "predicted_churn": (
            probabilities >= threshold
        ).astype(int),
    }).to_csv(
        REPORT_DIR / "explained_devices.csv",
        index=False,
    )

    # Save the fixed background for reproducible future explanations.
    background[features].to_parquet(
        EXPLAINER_DIR / "background_raw.parquet",
        index=False,
    )
    np.save(
        EXPLAINER_DIR / "background_transformed.npy",
        background_transformed,
    )

    metadata = {
        "release": "v1",
        "model_sha256": sha256(model_path),
        "shap_version": shap.__version__,
        "method": "LinearExplainer with independent masker",
        "explanation_units": "log_odds_of_inactivity",
        "background_partition": "train",
        "background_size": len(background),
        "explained_partition": "validation",
        "explained_devices": len(sample),
        "transformed_feature_names": transformed_names.tolist(),
        "limitations": [
            "Independent-feature attribution does not model correlations.",
            "Several behavioural features are correlated or redundant.",
            "Contributions explain predictions, not causes of churn.",
            "Plot feature values are transformed, not original units.",
        ],
    }

    (EXPLAINER_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Frozen model and policy hashes verified.")
    print(f"SHAP version: {shap.__version__}")
    print(f"Background devices: {len(background)}")
    print(f"Explained validation devices: {len(sample)}")
    print(f"Original features: {len(features)}")
    print(f"Transformed features: {len(transformed_names)}")
    print("Log-odds reconstruction check passed.")
    print("Probability reconstruction check passed.")

    print("\nGLOBAL FEATURE IMPORTANCE")
    print(importance.round(6).to_string(index=False))

    print("\nEXAMPLE DEVICE")
    print(f"Device: {sample.loc[row_index, 'device']}")
    print(f"Predicted inactivity: {probabilities[row_index]:.2%}")
    print(f"Classification threshold: {threshold:.2f}")

    print("\nTOP CONTRIBUTIONS")
    print(
        contributions[
            ["feature", "shap_log_odds", "direction"]
        ].head(5).round(6).to_string(index=False)
    )

    print(f"\nReports saved to: {REPORT_DIR}")
    print("No model fitting or threshold changes were performed.")


if __name__ == "__main__":
    main()