from pathlib import Path
import hashlib
import json

import psycopg
from psycopg.types.json import Jsonb

from check_database import connection_settings


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts/releases/v1/model.joblib"
POLICY_PATH = ROOT / "artifacts/releases/v1/decision_policy.json"
METRICS_PATH = ROOT / "reports/final_test/metrics.json"

SEGMENT_PATH = ROOT / "artifacts/segmentation/kmeans_pipeline.joblib"
SEGMENT_LABELS_PATH = ROOT / "artifacts/segmentation/segment_labels.json"

MODEL_VERSION = "v1"
SEGMENT_VERSION = "segments_v1"

SEGMENT_FEATURES = [
    "plays_7d",
    "active_days_7d",
    "days_since_last_play",
    "mean_score_7d",
]


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path):
    return path.relative_to(ROOT).as_posix()


def main():
    required = [
        MODEL_PATH,
        POLICY_PATH,
        METRICS_PATH,
        SEGMENT_PATH,
        SEGMENT_LABELS_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    policy = read_json(POLICY_PATH)
    metrics = read_json(METRICS_PATH)
    segment_labels = read_json(SEGMENT_LABELS_PATH)

    model_hash = sha256(MODEL_PATH)
    policy_hash = sha256(POLICY_PATH)
    segment_hash = sha256(SEGMENT_PATH)

    # Verify the exact artifacts used by earlier project stages.
    if model_hash != metrics["model_sha256"]:
        raise ValueError("Frozen model hash does not match final evaluation.")

    if policy_hash != metrics["policy_sha256"]:
        raise ValueError("Frozen policy hash does not match final evaluation.")

    if segment_hash != segment_labels["segmentation_sha256"]:
        raise ValueError("Segmentation hash does not match segment labels.")

    features = policy["features"]
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(name, str) for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("Invalid model feature list.")

    threshold = float(policy["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid decision threshold.")

    if threshold != float(metrics["threshold"]):
        raise ValueError("Policy and final evaluation thresholds differ.")

    names = segment_labels["names"]
    if (
        not isinstance(names, dict)
        or set(names) != {"0", "1"}
        or not all(isinstance(value, str) and value for value in names.values())
    ):
        raise ValueError("Expected names for segment IDs 0 and 1.")

    model_type = "unweighted_logistic_regression"
    model_relative = relative_path(MODEL_PATH)
    segment_relative = relative_path(SEGMENT_PATH)

    print("Frozen model and policy hashes verified.")
    print("Segmentation artifact hash verified.")
    print(f"Churn model version: {MODEL_VERSION}")
    print(f"Decision threshold: {threshold}")
    print(f"Segmentation version: {SEGMENT_VERSION}")

    settings = connection_settings()
    settings["application_name"] = "gamepulse_model_registration"

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (104004,))

            cursor.execute(
                """
                INSERT INTO gamepulse.model_versions (
                    model_version,
                    model_type,
                    artifact_path,
                    artifact_sha256,
                    feature_names,
                    decision_threshold,
                    evaluation_metrics
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_version) DO NOTHING
                """,
                (
                    MODEL_VERSION,
                    model_type,
                    model_relative,
                    model_hash,
                    Jsonb(features),
                    threshold,
                    Jsonb(metrics),
                ),
            )
            new_models = cursor.rowcount

            cursor.execute(
                """
                SELECT
                    model_type,
                    artifact_path,
                    artifact_sha256,
                    feature_names,
                    decision_threshold,
                    evaluation_metrics
                FROM gamepulse.model_versions
                WHERE model_version = %s
                """,
                (MODEL_VERSION,),
            )

            expected_model = (
                model_type,
                model_relative,
                model_hash,
                features,
                threshold,
                metrics,
            )

            if cursor.fetchone() != expected_model:
                raise ValueError(
                    "Stored model version has different contents. "
                    "Registration rolled back; existing version was not overwritten."
                )

            cursor.execute(
                """
                INSERT INTO gamepulse.segment_versions (
                    segment_version,
                    artifact_path,
                    artifact_sha256,
                    feature_names,
                    segment_names
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (segment_version) DO NOTHING
                """,
                (
                    SEGMENT_VERSION,
                    segment_relative,
                    segment_hash,
                    Jsonb(SEGMENT_FEATURES),
                    Jsonb(names),
                ),
            )
            new_segments = cursor.rowcount

            cursor.execute(
                """
                SELECT
                    artifact_path,
                    artifact_sha256,
                    feature_names,
                    segment_names
                FROM gamepulse.segment_versions
                WHERE segment_version = %s
                """,
                (SEGMENT_VERSION,),
            )

            expected_segment = (
                segment_relative,
                segment_hash,
                SEGMENT_FEATURES,
                names,
            )

            if cursor.fetchone() != expected_segment:
                raise ValueError(
                    "Stored segmentation version has different contents. "
                    "Registration rolled back; existing version was not overwritten."
                )

    # Printed only after the transaction commits successfully.
    print("\nREGISTRATION COMMITTED")
    print(f"New churn model versions: {new_models}")
    print(f"New segmentation versions: {new_segments}")
    print("Stored registry entries match the source artifacts.")
    print("Model files remain on disk; PostgreSQL stores their metadata.")
    print("No training, prediction generation, or threshold changes performed.")


if __name__ == "__main__":
    main()