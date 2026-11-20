from pathlib import Path
import hashlib

import psycopg

from check_database import connection_settings


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "sql" / "001_initial_schema.sql"
MIGRATION_NAME = "001_initial_schema"

EXPECTED_TABLES = {
    "schema_migrations",
    "players",
    "gameplay_events",
    "feature_snapshots",
    "model_versions",
    "segment_versions",
    "player_segments",
    "predictions",
    "observed_outcomes",
}


def main():
    sql_text = SQL_PATH.read_text(encoding="utf-8-sig")
    checksum = hashlib.sha256(
        sql_text.encode("utf-8")
    ).hexdigest()

    settings = connection_settings()
    settings["application_name"] = "gamepulse_schema_setup"

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            # Prevent two instances applying this migration concurrently.
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s);",
                (104001,),
            )

            cursor.execute("CREATE SCHEMA IF NOT EXISTS gamepulse;")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gamepulse.schema_migrations (
                    migration_name TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                SELECT checksum
                FROM gamepulse.schema_migrations
                WHERE migration_name = %s;
                """,
                (MIGRATION_NAME,),
            )
            existing = cursor.fetchone()

            if existing:
                if existing[0] != checksum:
                    raise ValueError(
                        "An applied migration has been changed. "
                        "Restore the original file; future schema changes "
                        "must use a new migration."
                    )
                status = "Migration already applied; no changes needed."
            else:
                # Execute our local, trusted schema file.
                cursor.execute(sql_text)

                cursor.execute(
                    """
                    INSERT INTO gamepulse.schema_migrations
                        (migration_name, checksum)
                    VALUES (%s, %s);
                    """,
                    (MIGRATION_NAME, checksum),
                )
                status = "Initial migration applied successfully."

            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """,
                ("gamepulse",),
            )

            tables = [row[0] for row in cursor.fetchall()]
            missing = EXPECTED_TABLES - set(tables)

            if missing:
                raise RuntimeError(
                    f"Expected tables are missing: {sorted(missing)}"
                )

    # The connection context commits before reaching this point.
    print(status)
    print("\nDATABASE TABLES")
    for table in tables:
        print(f"- gamepulse.{table}")

    print(f"\nExpected tables verified: {len(EXPECTED_TABLES)}")
    print("Schema transaction completed.")
    print("No gameplay data or predictions have been imported yet.")


if __name__ == "__main__":
    main()