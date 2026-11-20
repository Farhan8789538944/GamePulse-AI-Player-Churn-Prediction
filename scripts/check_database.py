from pathlib import Path

import psycopg
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env.database"


def connection_settings():
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            "Missing .env.database in the project root."
        )

    config = dotenv_values(ENV_PATH, encoding="utf-8-sig")

    required = [
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_PORT",
    ]

    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(
            f"Missing database settings: {', '.join(missing)}"
        )

    return {
        "host": "127.0.0.1",
        "port": int(config["POSTGRES_PORT"]),
        "dbname": config["POSTGRES_DB"],
        "user": config["POSTGRES_USER"],
        "password": config["POSTGRES_PASSWORD"],
        "connect_timeout": 10,
        "application_name": "gamepulse_connection_check",
    }


def main():
    settings = connection_settings()

    # Credentials are passed separately, never printed.
    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_database(),
                       current_user,
                       current_setting('server_version');
                """
            )
            database, username, version = cursor.fetchone()

            # Verify a parameterised query.
            cursor.execute(
                "SELECT %s::integer + %s::integer;",
                (20, 22),
            )
            result = cursor.fetchone()[0]

            if result != 42:
                raise AssertionError("Query verification failed.")

    print("Python-to-PostgreSQL connection successful.")
    print(f"Database: {database}")
    print(f"User: {username}")
    print(f"Server version: {version}")
    print(f"Connection: {settings['host']}:{settings['port']}")
    print("Parameterised query check passed.")
    print("Connection closed successfully.")
    print("No application tables or records were changed.")


if __name__ == "__main__":
    main()