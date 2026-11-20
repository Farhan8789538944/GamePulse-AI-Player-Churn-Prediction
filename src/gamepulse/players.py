import logging

import psycopg
from fastapi import APIRouter, HTTPException, Query

from gamepulse.database import read_connection


router = APIRouter(prefix="/v1/players", tags=["Players"])
logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"
FEATURE_VERSION = "features_v1"
RULES_VERSION = "rules_v1"
PARTITION = "validation"

# Fixed SQL controlled by the application.
BASE_QUERY = """
    SELECT
        s.snapshot_id,
        s.device_id,
        s.cutoff AS prediction_cutoff,
        s.data_partition,
        p.model_version,
        m.decision_threshold,
        p.churn_probability,
        p.predicted_churn,
        p.rules_version,
        p.recommendation
    FROM gamepulse.feature_snapshots s
    JOIN gamepulse.predictions p
      ON p.snapshot_id = s.snapshot_id
    JOIN gamepulse.model_versions m
      ON m.model_version = p.model_version
    WHERE s.feature_version = %s
      AND s.data_partition = %s
      AND p.model_version = %s
      AND p.rules_version = %s
"""

BASE_PARAMS = (
    FEATURE_VERSION,
    PARTITION,
    MODEL_VERSION,
    RULES_VERSION,
)


def database_error(error):
    logger.error("Player query failed: %s", type(error).__name__)
    return HTTPException(
        status_code=503,
        detail="Player data is temporarily unavailable.",
    )


def format_player(row):
    result = dict(row)
    result["at_or_above_threshold"] = bool(result.pop("predicted_churn"))
    return result


@router.get("")
def list_players(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Browse stored validation predictions, highest inactivity risk first."""
    try:
        with read_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    BASE_QUERY
                    + """
                    ORDER BY p.churn_probability DESC, s.snapshot_id ASC
                    LIMIT %s OFFSET %s
                    """,
                    (*BASE_PARAMS, limit + 1, offset),
                )
                rows = cursor.fetchall()

        has_more = len(rows) > limit
        items = [format_player(row) for row in rows[:limit]]

        return {
            "partition": PARTITION,
            "limit": limit,
            "offset": offset,
            "returned": len(items),
            "has_more": has_more,
            "next_offset": offset + limit if has_more else None,
            "items": items,
        }

    except (psycopg.Error, RuntimeError, ValueError) as error:
        raise database_error(error) from error


@router.get("/{device_id}")
def get_player(device_id: str):
    """Retrieve a device's stored validation prediction and input features."""
    try:
        with read_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    BASE_QUERY + " AND s.device_id = %s",
                    (*BASE_PARAMS, device_id),
                )
                row = cursor.fetchone()

                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            "No scored validation snapshot found for this device."
                        ),
                    )

                cursor.execute(
                    """
                    SELECT features
                    FROM gamepulse.feature_snapshots
                    WHERE snapshot_id = %s
                    """,
                    (row["snapshot_id"],),
                )
                features = cursor.fetchone()["features"]

        result = format_player(row)
        result["features"] = features
        result["interpretation"] = (
            "Device IDs are player proxies. Predictions concern 14-day "
            "inactivity, not permanent churn. Retention suggestions have "
            "not been proven to improve retention."
        )
        return result

    except (psycopg.Error, RuntimeError, ValueError) as error:
        raise database_error(error) from error