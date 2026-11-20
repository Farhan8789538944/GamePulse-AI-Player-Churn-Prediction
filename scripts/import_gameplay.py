from pathlib import Path

import pandas as pd
import psycopg

from check_database import connection_settings


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "game1_clean_events.parquet"


def main():
    events = pd.read_parquet(DATA_PATH)

    required_columns = {"device", "event_time", "score"}
    missing_columns = required_columns - set(events.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    events = events[["device", "event_time", "score"]].copy()
    events["device"] = events["device"].astype("string").str.strip()
    events["event_time"] = pd.to_datetime(
        events["event_time"], utc=True, errors="raise"
    )
    events["score"] = pd.to_numeric(events["score"], errors="raise")

    if events.empty:
        raise ValueError("The cleaned event file is empty.")

    if events.isna().any().any():
        raise ValueError("Cleaned events contain missing values.")

    if events["device"].eq("").any():
        raise ValueError("Cleaned events contain empty device IDs.")

    if (
        (events["score"] < 0).any()
        or (events["score"] % 1 != 0).any()
    ):
        raise ValueError("Scores must be nonnegative integers.")

    if events.duplicated(
        subset=["device", "event_time", "score"]
    ).any():
        raise ValueError("Cleaned data still contains duplicate events.")

    expected_events = len(events)
    expected_players = events["device"].nunique()

    print(f"Source players: {expected_players:,}")
    print(f"Source events: {expected_events:,}")
    print("Importing cleaned historical gameplay data...")

    settings = connection_settings()
    settings["application_name"] = "gamepulse_gameplay_import"

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            # Prevent concurrent runs of this import.
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s);",
                (104002,),
            )

            cursor.execute(
                """
                CREATE TEMP TABLE staged_gameplay (
                    device_id TEXT NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    score BIGINT NOT NULL
                ) ON COMMIT DROP;
                """
            )

            # COPY efficiently transfers rows to PostgreSQL.
            with cursor.copy(
                """
                COPY staged_gameplay (device_id, event_time, score)
                FROM STDIN
                """
            ) as copy:
                for device, event_time, score in events.itertuples(
                    index=False, name=None
                ):
                    copy.write_row((
                        str(device),
                        event_time.to_pydatetime(),
                        int(score),
                    ))

            cursor.execute(
                """
                INSERT INTO gamepulse.players (
                    device_id,
                    first_observed_at
                )
                SELECT device_id, MIN(event_time)
                FROM staged_gameplay
                GROUP BY device_id
                ON CONFLICT (device_id) DO NOTHING;
                """
            )
            inserted_players = cursor.rowcount

            # Detect conflicting first-observed timestamps rather than
            # silently overwriting existing player metadata.
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT device_id, MIN(event_time) AS first_event
                    FROM staged_gameplay
                    GROUP BY device_id
                ) AS source
                JOIN gamepulse.players AS stored
                    ON stored.device_id = source.device_id
                WHERE stored.first_observed_at
                    IS DISTINCT FROM source.first_event;
                """
            )

            if cursor.fetchone()[0] != 0:
                raise ValueError(
                    "Existing player timestamps differ from this source. "
                    "Import rolled back; investigate before continuing."
                )

            cursor.execute(
                """
                INSERT INTO gamepulse.gameplay_events (
                    device_id,
                    event_time,
                    score
                )
                SELECT device_id, event_time, score
                FROM staged_gameplay
                ON CONFLICT (device_id, event_time, score)
                DO NOTHING;
                """
            )
            inserted_events = cursor.rowcount

            # Verify that every source event exists in the database.
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM staged_gameplay AS source
                JOIN gamepulse.gameplay_events AS stored
                  ON stored.device_id = source.device_id
                 AND stored.event_time = source.event_time
                 AND stored.score = source.score;
                """
            )
            matched_events = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(DISTINCT source.device_id)
                FROM staged_gameplay AS source
                JOIN gamepulse.players AS stored
                  ON stored.device_id = source.device_id;
                """
            )
            matched_players = cursor.fetchone()[0]

            if matched_events != expected_events:
                raise RuntimeError("Imported event verification failed.")

            if matched_players != expected_players:
                raise RuntimeError("Imported player verification failed.")

            cursor.execute("SELECT COUNT(*) FROM gamepulse.players;")
            total_players = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM gamepulse.gameplay_events;"
            )
            total_events = cursor.fetchone()[0]

    # Successful exit from the connection context commits the transaction.
    print("\nIMPORT COMMITTED")
    print(f"New players inserted: {inserted_players:,}")
    print(f"New events inserted: {inserted_events:,}")

    print("\nSOURCE VERIFICATION")
    print(f"Players matched: {matched_players:,}")
    print(f"Events matched: {matched_events:,}")

    print("\nDATABASE TOTALS")
    print(f"Players: {total_players:,}")
    print(f"Gameplay events: {total_events:,}")

    print("\nRaw and processed files were not modified.")
    print("No models were trained or predictions generated.")


if __name__ == "__main__":
    main()