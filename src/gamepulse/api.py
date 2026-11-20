import logging

import psycopg
from fastapi import FastAPI, HTTPException

from gamepulse.database import read_connection


logger = logging.getLogger(__name__)

app = FastAPI(
    title="GamePulse AI",
    version="0.1.0",
    description=(
        "Historical player inactivity predictions, "
        "engagement segments, and analyst retention suggestions."
    ),
)

MODEL_VERSION = "v1"
FEATURE_VERSION = "features_v1"
RULES_VERSION = "rules_v1"
PARTITION = "validation"


def database_unavailable(error):
    # Avoid exposing credentials or database internals in responses.
    logger.error("Database request failed: %s", type(error).__name__)
    return HTTPException(
        status_code=503,
        detail="Database unavailable or request could not be completed.",
    )


@app.get("/health/live")
def liveness():
    """Check whether the API process is responding."""
    return {
        "status": "ok",
        "service": "gamepulse-api",
    }


@app.get("/health/ready")
def readiness():
    """Check database access and the required registered model."""
    try:
        with read_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT model_version
                    FROM gamepulse.model_versions
                    WHERE model_version = %s
                    """,
                    (MODEL_VERSION,),
                )
                model = cursor.fetchone()

        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Required model version is not registered.",
            )

        return {
            "status": "ready",
            "database": "reachable",
            "model_version": model["model_version"],
        }

    except (psycopg.Error, RuntimeError, ValueError) as error:
        raise database_unavailable(error) from error


@app.get("/v1/summary")
def validation_summary():
    """Summarize stored validation predictions without reading outcomes."""
    try:
        with read_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT decision_threshold
                    FROM gamepulse.model_versions
                    WHERE model_version = %s
                    """,
                    (MODEL_VERSION,),
                )
                model = cursor.fetchone()

                if model is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Required model version is not registered.",
                    )

                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS snapshots,
                        COUNT(p.prediction_id) AS scored_snapshots,
                        COUNT(*) FILTER (
                            WHERE p.predicted_churn = 1
                        ) AS at_or_above_threshold,
                        COUNT(*) FILTER (
                            WHERE p.predicted_churn = 0
                        ) AS below_threshold,
                        AVG(p.churn_probability) AS mean_predicted_inactivity
                    FROM gamepulse.feature_snapshots s
                    LEFT JOIN gamepulse.predictions p
                      ON p.snapshot_id = s.snapshot_id
                     AND p.model_version = %s
                     AND p.rules_version = %s
                    WHERE s.feature_version = %s
                      AND s.data_partition = %s
                    """,
                    (
                        MODEL_VERSION,
                        RULES_VERSION,
                        FEATURE_VERSION,
                        PARTITION,
                    ),
                )
                summary = cursor.fetchone()

        return {
            "partition": PARTITION,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "rules_version": RULES_VERSION,
            "decision_threshold": model["decision_threshold"],
            **summary,
            "unscored_snapshots": (
                summary["snapshots"] - summary["scored_snapshots"]
            ),
            "interpretation": (
                "Predictions concern inactivity during the 14-day outcome "
                "window. Below the decision threshold does not necessarily "
                "mean low risk. These are historical validation records."
            ),
        }

    except (psycopg.Error, RuntimeError, ValueError) as error:
        raise database_unavailable(error) from error

from gamepulse.players import router as players_router

app.include_router(players_router)

from gamepulse.predict import router as predict_router

app.include_router(predict_router)

from gamepulse.performance import router as performance_router

app.include_router(performance_router)

from gamepulse.explanations import router as explanations_router

app.include_router(explanations_router)