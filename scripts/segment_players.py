from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "segmentation"
ARTIFACT_DIR = ROOT / "artifacts" / "segmentation"

SEGMENT_FEATURES = [
    "plays_7d",
    "active_days_7d",
    "days_since_last_play",
    "mean_score_7d",
]

K_VALUES = [2, 3, 4, 5]


def profile_segments(data, assignments):
    """Summarise behaviour in original, interpretable units."""
    frame = data[SEGMENT_FEATURES].copy()
    frame["segment_id"] = assignments

    profile = frame.groupby("segment_id").agg(
        devices=("plays_7d", "size"),
        median_plays_7d=("plays_7d", "median"),
        median_active_days=("active_days_7d", "median"),
        median_days_since_last_play=(
            "days_since_last_play", "median"
        ),
        median_mean_score=("mean_score_7d", "median"),
    )

    profile["share_percent"] = (
        profile["devices"] / len(frame) * 100
    )

    return profile


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(
        DATA_DIR / "game1_train.parquet"
    )
    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )

    assert set(train["device"]).isdisjoint(validation["device"])

    X_train = train[SEGMENT_FEATURES]
    X_validation = validation[SEGMENT_FEATURES]

    for name, frame in [
        ("training", X_train),
        ("validation", X_validation),
    ]:
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{name} features contain missing/invalid values.")

        if (frame < 0).any().any():
            raise ValueError(f"{name} features must be nonnegative.")

    preprocessing = Pipeline([
        (
            "log_transform",
            FunctionTransformer(
                np.log1p,
                feature_names_out="one-to-one",
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    # No churn labels are supplied to preprocessing or clustering.
    transformed_train = preprocessing.fit_transform(X_train)

    # Use the same random training sample for all silhouette comparisons.
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(train),
        size=min(2000, len(train)),
        replace=False,
    )

    candidates = {}
    results = []

    print(f"Training devices: {len(train):,}")
    print(f"Validation devices: {len(validation):,}")
    print("Test data: NOT LOADED")
    print("Churn labels are NOT used to select clusters.")

    for k in K_VALUES:
        print(f"\nFitting K-Means with k={k}")

        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42,
        )
        assignments = model.fit_predict(transformed_train)

        sample_labels = assignments[sample_indices]
        if len(np.unique(sample_labels)) < 2:
            raise ValueError(
                f"Silhouette sample has fewer than two clusters for k={k}."
            )

        silhouette = silhouette_score(
            transformed_train[sample_indices],
            sample_labels,
        )

        sizes = np.bincount(assignments, minlength=k)

        results.append({
            "k": k,
            "sampled_silhouette": float(silhouette),
            "inertia": float(model.inertia_),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
        })

        candidates[k] = model

    comparison = pd.DataFrame(results).sort_values(
        ["sampled_silhouette", "k"],
        ascending=[False, True],
    )

    comparison.to_csv(
        REPORT_DIR / "cluster_comparison.csv",
        index=False,
    )

    best_k = int(comparison.iloc[0]["k"])
    selected_model = candidates[best_k]

    # Assemble fitted components; no additional fit is required.
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("kmeans", selected_model),
    ])

    train_assignments = pipeline.predict(X_train)
    validation_assignments = pipeline.predict(X_validation)

    model_path = ARTIFACT_DIR / "kmeans_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    loaded = joblib.load(model_path)
    np.testing.assert_array_equal(
        pipeline.predict(X_validation.iloc[:100]),
        loaded.predict(X_validation.iloc[:100]),
    )

    train_profile = profile_segments(
        train, train_assignments
    )
    validation_profile = profile_segments(
        validation, validation_assignments
    )

    # Outcomes are attached only AFTER clustering has been selected.
    validation_outcomes = pd.DataFrame({
        "segment_id": validation_assignments,
        "churn": validation["churn"].to_numpy(),
    }).groupby("segment_id").agg(
        inactive_devices=("churn", "sum"),
        observed_churn_rate=("churn", "mean"),
    )

    validation_profile = validation_profile.join(
        validation_outcomes
    )
    validation_profile["returned_devices"] = (
        validation_profile["devices"]
        - validation_profile["inactive_devices"]
    )
    validation_profile["observed_churn_percent"] = (
        validation_profile.pop("observed_churn_rate") * 100
    )

    train_profile.to_csv(REPORT_DIR / "train_profiles.csv")
    validation_profile.to_csv(
        REPORT_DIR / "validation_profiles.csv"
    )

    for partition, data, assignments in [
        ("train", train, train_assignments),
        ("validation", validation, validation_assignments),
    ]:
        pd.DataFrame({
            "device": data["device"].to_numpy(),
            "segment_id": assignments,
        }).to_csv(
            REPORT_DIR / f"{partition}_assignments.csv",
            index=False,
        )

    metadata = {
        "features": SEGMENT_FEATURES,
        "selected_k": best_k,
        "selection_metric": "sampled_training_silhouette",
        "silhouette_sample_size": len(sample_indices),
        "candidate_k_values": K_VALUES,
        "random_seed": 42,
        "preprocessing": "log1p followed by StandardScaler",
        "training_devices": len(train),
        "cluster_names": {
            str(i): f"Segment {i}" for i in range(best_k)
        },
        "limitations": [
            "Clusters are descriptive, not ground-truth player types.",
            "Cluster IDs are arbitrary.",
            "Silhouette selection does not establish cluster stability.",
            "Scores reflect observed gameplay, not verified player skill.",
            "Device IDs are proxies for players.",
        ],
    }

    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCLUSTER COMPARISON")
    print(comparison.round(4).to_string(index=False))

    print(f"\nSelected number of clusters: {best_k}")

    print("\nTRAINING SEGMENT PROFILES")
    print(train_profile.round(2).to_string())

    print("\nVALIDATION SEGMENT PROFILES")
    print(validation_profile.round(2).to_string())

    print("\nSaved-model prediction check passed.")
    print(f"Pipeline saved to: {model_path}")
    print(f"Reports saved to: {REPORT_DIR}")
    print("Churn model v1 and its threshold remain unchanged.")


if __name__ == "__main__":
    main()