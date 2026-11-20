from pathlib import Path
import json
import math

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from check_database import connection_settings


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FEATURE_VERSION = "features_v1"
LABEL_VERSION = "inactivity_7d_14d_v1"


def load_source():
    policy = json.loads(
        (
            ROOT / "artifacts" / "releases" / "v1"
            / "decision_policy.json"
        ).read_text(encoding="utf-8")
    )
    features = policy["features"]

    frames = []

    for partition in ["train", "validation", "test"]:
        frame = pd.read_parquet(
            DATA_DIR / f"game1_{partition}.parquet"
        ).copy()
        frame["data_partition"] = partition
        frames.append(frame)

    source = pd.concat(frames, ignore_index=True)

    if source.empty or source["device"].duplicated().any():
        raise ValueError("Source is empty or contains overlapping devices.")

    labels = pd.read_csv(
        DATA_DIR / "game1_labels.csv",
        dtype={"device": "string"},
    )

    # Validate that the outcome information matches our partition files.
    for column in ["first_event", "cutoff", "outcome_end"]:
        labels[column] = pd.to_datetime(labels[column], utc=True)
        source[column] = pd.to_datetime(source[column], utc=True)

    source["device"] = source["device"].astype("string")

    joined = source.merge(
        labels[
            [
                "device",
                "first_event",
                "cutoff",
                "outcome_end",
                "churn",
                "outcome_event_count",
            ]
        ],
        on="device",
        how="left",
        suffixes=("", "_label"),
        validate="one_to_one",
        indicator=True,
    )

    if not joined["_merge"].eq("both").all():
        raise ValueError("Some snapshots have no matching label record.")

    for column in ["first_event", "cutoff", "outcome_end", "churn"]:
        if not joined[column].eq(joined[f"{column}_label"]).all():
            raise ValueError(f"Label mismatch in {column}.")

    if not (
        joined["cutoff"] - joined["first_event"]
    ).eq(pd.Timedelta(days=7)).all():
        raise ValueError("Unexpected observation window.")

    if not (
        joined["outcome_end"] - joined["cutoff"]
    ).eq(pd.Timedelta(days=14)).all():
        raise ValueError("Unexpected outcome window.")

    counts = joined["outcome_event_count"]

    if (
        counts.isna().any()
        or (counts < 0).any()
        or (counts % 1 != 0).any()
    ):
        raise ValueError("Invalid outcome event counts.")

    if not joined["churn"].eq(counts.eq(0).astype(int)).all():
        raise ValueError("Outcome count and churn label disagree.")

    return joined, features


def feature_payload(row, feature_names):
    payload = {}

    for name in feature_names:
        value = row[name]

        if pd.isna(value):
            if name not in {"mean_gap_hours", "max_gap_hours"}:
                raise ValueError(f"Unexpected missing feature: {name}")
            payload[name] = None
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Non-finite feature: {name}")
            payload[name] = number

    return payload


def main():
    source, features = load_source()

    print(f"Source snapshots: {len(source):,}")
    print("\nSOURCE PARTITIONS")
    print(source["data_partition"].value_counts().to_string())

    settings = connection_settings()
    settings["application_name"] = "gamepulse_snapshot_import"

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s);",
                (104003,),
            )

            cursor.execute(
                """
                CREATE TEMP TABLE staged_snapshots (
                    device_id TEXT PRIMARY KEY,
                    first_event TIMESTAMPTZ NOT NULL,
                    cutoff TIMESTAMPTZ NOT NULL,
                    feature_version TEXT NOT NULL,
                    features JSONB NOT NULL,
                    data_partition TEXT NOT NULL,
                    label_version TEXT NOT NULL,
                    outcome_end TIMESTAMPTZ NOT NULL,
                    outcome_event_count INTEGER NOT NULL,
                    churn SMALLINT NOT NULL
                ) ON COMMIT DROP;
                """
            )

            with cursor.copy(
                """
                COPY staged_snapshots (
                    device_id, first_event, cutoff,
                    feature_version, features, data_partition,
                    label_version, outcome_end,
                    outcome_event_count, churn
                ) FROM STDIN
                """
            ) as copy:
                for _, row in source.iterrows():
                    copy.write_row((
                        str(row["device"]),
                        row["first_event"].to_pydatetime(),
                        row["cutoff"].to_pydatetime(),
                        FEATURE_VERSION,
                        Jsonb(feature_payload(row, features)),
                        str(row["data_partition"]),
                        LABEL_VERSION,
                        row["outcome_end"].to_pydatetime(),
                        int(row["outcome_event_count"]),
                        int(row["churn"]),
                    ))

            cursor.execute(
                """
                INSERT INTO gamepulse.feature_snapshots (
                    device_id, first_event, cutoff,
                    feature_version, features, data_partition
                )
                SELECT device_id, first_event, cutoff,
                       feature_version, features, data_partition
                FROM staged_snapshots
                ON CONFLICT (device_id, cutoff, feature_version)
                DO NOTHING;
                """
            )
            inserted_snapshots = cursor.rowcount

            # Existing rows must match; never silently overwrite features.
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM staged_snapshots AS source
                JOIN gamepulse.feature_snapshots AS stored
                  ON stored.device_id = source.device_id
                 AND stored.cutoff = source.cutoff
                 AND stored.feature_version = source.feature_version
                WHERE stored.first_event = source.first_event
                  AND stored.features = source.features
                  AND stored.data_partition = source.data_partition;
                """
            )
            matched_snapshots = cursor.fetchone()[0]

            if matched_snapshots != len(source):
                raise ValueError(
                    "Stored snapshots differ from source. Import rolled back."
                )

            cursor.execute(
                """
                INSERT INTO gamepulse.observed_outcomes (
                    snapshot_id, label_version,
                    outcome_start, outcome_end,
                    outcome_event_count, churn, label_available_at
                )
                SELECT stored.snapshot_id, source.label_version,
                       source.cutoff, source.outcome_end,
                       source.outcome_event_count, source.churn,
                       source.outcome_end
                FROM staged_snapshots AS source
                JOIN gamepulse.feature_snapshots AS stored
                  ON stored.device_id = source.device_id
                 AND stored.cutoff = source.cutoff
                 AND stored.feature_version = source.feature_version
                ON CONFLICT (snapshot_id, label_version)
                DO NOTHING;
                """
            )
            inserted_outcomes = cursor.rowcount

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM staged_snapshots AS source
                JOIN gamepulse.feature_snapshots AS snapshot
                  ON snapshot.device_id = source.device_id
                 AND snapshot.cutoff = source.cutoff
                 AND snapshot.feature_version = source.feature_version
                JOIN gamepulse.observed_outcomes AS outcome
                  ON outcome.snapshot_id = snapshot.snapshot_id
                 AND outcome.label_version = source.label_version
                WHERE outcome.outcome_start = source.cutoff
                  AND outcome.outcome_end = source.outcome_end
                  AND outcome.outcome_event_count = source.outcome_event_count
                  AND outcome.churn = source.churn
                  AND outcome.label_available_at = source.outcome_end;
                """
            )
            matched_outcomes = cursor.fetchone()[0]

            if matched_outcomes != len(source):
                raise ValueError(
                    "Stored outcomes differ from source. Import rolled back."
                )

            cursor.execute(
                """
                SELECT data_partition, COUNT(*)
                FROM gamepulse.feature_snapshots
                WHERE feature_version = %s
                GROUP BY data_partition
                ORDER BY data_partition;
                """,
                (FEATURE_VERSION,),
            )
            partition_counts = cursor.fetchall()

    print("\nIMPORT COMMITTED")
    print(f"New snapshots inserted: {inserted_snapshots:,}")
    print(f"New outcomes inserted: {inserted_outcomes:,}")

    print("\nSOURCE VERIFICATION")
    print(f"Snapshots matched: {matched_snapshots:,}")
    print(f"Outcomes matched: {matched_outcomes:,}")

    print("\nDATABASE SNAPSHOTS FOR features_v1")
    for partition, count in partition_counts:
        print(f"{partition}: {count:,}")

    print("\nMissing gap features were stored as JSON null.")
    print("Future outcomes are separate from model input features.")
    print("No model fitting or prediction changes were performed.")


if __name__ == "__main__":
    main()