from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row


ROOT = Path(__file__).resolve().parents[2]


def connection_settings():
    values = dotenv_values(
        ROOT / ".env.database",
        encoding="utf-8-sig",
    )

    required = [
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_PORT",
    ]

    if any(not values.get(name) for name in required):
        raise RuntimeError("Database configuration is incomplete.")

    return {
        "host": "127.0.0.1",
        "port": int(values["POSTGRES_PORT"]),
        "dbname": values["POSTGRES_DB"],
        "user": values["POSTGRES_USER"],
        "password": values["POSTGRES_PASSWORD"],
        "connect_timeout": 5,
        "application_name": "gamepulse_api",
    }


@contextmanager
def read_connection():
    with psycopg.connect(
        **connection_settings(),
        row_factory=dict_row,
    ) as connection:
        # These initial endpoints only read stored data.
        connection.read_only = True

        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '5s'")

        yield connection