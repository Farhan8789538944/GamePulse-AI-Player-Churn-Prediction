from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from check_database import connection_settings
from recommend_retention import recommend, RULE_CONFIG


ROOT = Path(__file__).resolve().parents[1]

MODEL_VERSION = "v1"
SEGMENT_VERSION = "segments_v1"
FEATURE_VERSION = "features_v1"
PARTITION = "validation"

POLICY_PATH = ROOT / "artifacts/releases/v1/decision_policy.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_artifact(relative_path, expected_hash):
    path = (ROOT / relative_path).resolve()

    if not path.is_relative_to(ROOT):
        raise ValueError("Artifact path must stay inside the project.")

    if not path.is_file():
        raise FileNotFoundError(path)

    if sha256(path) != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {path.name}")

    # Load only our own verified project artifacts.
    return joblib.load(path)


def main():
    rules_version = RULE_CONFIG["version"]
    settings = connection_settings()
    settings["application_name"] = "gamepulse_validation_scoring"

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (104005,))

            cursor.execute(
                """
                SELECT artifact_path, artifact_sha256, feature_names,
                       decision_threshold, evaluation_metrics
                FROM gamepulse.model_versions
                WHERE model_version = %s
                """,
                (MODEL_VERSION,),
            )
            model_record = cursor.fetchone()
            if model_record is None:
                raise ValueError("Register the churn model first.")

            (
                model_path,
                model_hash,
                feature_names,
                threshold,
                evaluation_metrics,
            ) = model_record

            with POLICY_PATH.open("r", encoding="utf-8-sig") as handle:
                policy = json.load(handle)

            if sha256(POLICY_PATH) != evaluation_metrics["policy_sha256"]:
                raise ValueError("Frozen decision policy hash mismatch.")

            if (
                policy["features"] != feature_names
                or float(policy["threshold"]) != threshold
            ):
                raise ValueError("Database metadata differs from frozen policy.")

            cursor.execute(
                """
                SELECT artifact_path, artifact_sha256,
                       feature_names, segment_names
                FROM gamepulse.segment_versions
                WHERE segment_version = %s
                """,
                (SEGMENT_VERSION,),
            )
            segment_record = cursor.fetchone()
            if segment_record is None:
                raise ValueError("Register the segmentation model first.")

            (
                segment_path,
                segment_hash,
                segment_features,
                segment_names,
            ) = segment_record

            model = load_verified_artifact(model_path, model_hash)
            segment_model = load_verified_artifact(segment_path, segment_hash)

            # Only observation-window features are read.
            # No observed outcomes are queried.
            cursor.execute(
                """
                SELECT snapshot_id, features
                FROM gamepulse.feature_snapshots
                WHERE feature_version = %s
                  AND data_partition = %s
                ORDER BY snapshot_id
                """,
                (FEATURE_VERSION, PARTITION),
            )
            rows = cursor.fetchall()

            if not rows:
                raise ValueError("No validation snapshots found.")

            snapshot_ids = [row[0] for row in rows]
            payloads = [row[1] for row in rows]

            for payload in payloads:
                if set(payload) != set(feature_names):
                    raise ValueError("Snapshot feature names do not match model.")

            X = pd.DataFrame(payloads, columns=feature_names).astype(float)

            allowed_missing = {"mean_gap_hours", "max_gap_hours"}
            for name in feature_names:
                if name not in allowed_missing and X[name].isna().any():
                    raise ValueError(f"Unexpected missing feature: {name}")

            if np.isinf(X.to_numpy()).any():
                raise ValueError("Infinite feature values found.")

            if not set(segment_features).issubset(X.columns):
                raise ValueError("Segmentation features are missing.")

            if X[segment_features].isna().any().any():
                raise ValueError("Segmentation features contain missing values.")

            classes = list(model.classes_)
            if set(classes) != {0, 1}:
                raise ValueError("Expected model classes 0 and 1.")

            probabilities = model.predict_proba(X)[:, classes.index(1)]

            if (
                not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError("Invalid predicted probabilities.")

            segments = segment_model.predict(X[segment_features])
            predicted_labels = (probabilities >= threshold).astype(int)

            print(f"Partition: {PARTITION}")
            print(f"Snapshots: {len(rows):,}")
            print(f"Model: {MODEL_VERSION}")
            print(f"Fixed threshold: {threshold}")
            print("Artifact and policy hashes verified.")

            recommendations = []
            for index, (_, features) in enumerate(X.iterrows()):
                segment_id = int(segments[index])
                segment_name = segment_names.get(str(segment_id))

                if segment_name is None:
                    raise ValueError(f"Unknown segment ID: {segment_id}")

                result = recommend(
                    plays_7d=float(features["plays_7d"]),
                    days_since_last_play=float(features["days_since_last_play"]),
                    churn_probability=float(probabilities[index]),
                    threshold=float(threshold),
                    config=RULE_CONFIG,
                )

                # Store rule settings and segment provenance with each suggestion.
                result = dict(result)
                result.update(
                    {
                        "rules_version": rules_version,
                        "rules_config": dict(RULE_CONFIG),
                        "segment_version": SEGMENT_VERSION,
                        "segment_id": segment_id,
                        "segment_name": segment_name,
                    }
                )
                recommendations.append(result)

            cursor.execute(
                """
                CREATE TEMP TABLE staged_scores (
                    snapshot_id BIGINT PRIMARY KEY,
                    probability DOUBLE PRECISION NOT NULL,
                    predicted_churn SMALLINT NOT NULL,
                    segment_id INTEGER NOT NULL,
                    recommendation JSONB NOT NULL
                ) ON COMMIT DROP
                """
            )

            with cursor.copy(
                """
                COPY staged_scores (
                    snapshot_id, probability, predicted_churn,
                    segment_id, recommendation
                ) FROM STDIN
                """
            ) as copy:
                for index, snapshot_id in enumerate(snapshot_ids):
                    copy.write_row(
                        (
                            int(snapshot_id),
                            float(probabilities[index]),
                            int(predicted_labels[index]),
                            int(segments[index]),
                            Jsonb(recommendations[index]),
                        )
                    )

            cursor.execute(
                """
                INSERT INTO gamepulse.player_segments (
                    snapshot_id, segment_version, segment_id
                )
                SELECT snapshot_id, %s, segment_id
                FROM staged_scores
                ON CONFLICT (snapshot_id, segment_version) DO NOTHING
                """,
                (SEGMENT_VERSION,),
            )
            new_segments = cursor.rowcount

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM staged_scores s
                JOIN gamepulse.player_segments p
                  ON p.snapshot_id = s.snapshot_id
                 AND p.segment_version = %s
                WHERE p.segment_id = s.segment_id
                """,
                (SEGMENT_VERSION,),
            )
            if cursor.fetchone()[0] != len(rows):
                raise ValueError("Stored segments differ; transaction rolled back.")

            cursor.execute(
                """
                INSERT INTO gamepulse.predictions (
                    snapshot_id, model_version, churn_probability,
                    predicted_churn, rules_version, recommendation
                )
                SELECT snapshot_id, %s, probability,
                       predicted_churn, %s, recommendation
                FROM staged_scores
                ON CONFLICT (
                    snapshot_id, model_version, rules_version
                ) DO NOTHING
                """,
                (MODEL_VERSION, rules_version),
            )
            new_predictions = cursor.rowcount

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM staged_scores s
                JOIN gamepulse.predictions p
                  ON p.snapshot_id = s.snapshot_id
                 AND p.model_version = %s
                 AND p.rules_version = %s
                WHERE ABS(p.churn_probability - s.probability) <= 1e-12
                  AND p.predicted_churn = s.predicted_churn
                  AND p.recommendation = s.recommendation
                """,
                (MODEL_VERSION, rules_version),
            )
            matched = cursor.fetchone()[0]

            if matched != len(rows):
                raise ValueError(
                    "Stored predictions or recommendations differ. "
                    "Transaction rolled back; existing results were not overwritten."
                )

    print("\nBATCH SCORING COMMITTED")
    print(f"New player segments: {new_segments:,}")
    print(f"New predictions: {new_predictions:,}")
    print(f"Stored predictions verified: {matched:,}")

    print("\nDECISIONS")
    print(f"At or above threshold: {int(predicted_labels.sum()):,}")
    print(f"Below threshold: {int((predicted_labels == 0).sum()):,}")

    print("\nRECOMMENDATION COUNTS")
    print(pd.Series(
        [item["rule_id"] for item in recommendations]
    ).value_counts().to_string())

    print("\nNo future outcomes were read.")
    print("No models were fitted or thresholds changed.")
    print("Recommendations are suggestions, not proven retention treatments.")


if __name__ == "__main__":
    main()