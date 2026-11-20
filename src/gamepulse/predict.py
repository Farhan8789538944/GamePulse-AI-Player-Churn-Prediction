import hashlib
import io
import json
import logging
from functools import lru_cache
from pathlib import Path
from gamepulse.engagement import engagement_details
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "artifacts/releases/v1"
METRICS_PATH = ROOT / "reports/final_test/metrics.json"

router = APIRouter(prefix="/v1", tags=["Predictions"])
logger = logging.getLogger(__name__)


class GameplayFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    plays_7d: int = Field(ge=1)
    active_days_7d: int = Field(ge=1, le=7)
    days_since_last_play: float = Field(gt=0, le=7)
    activity_span_days: float = Field(ge=0, lt=7)
    plays_first_3d: int = Field(ge=1)
    plays_last_3d: int = Field(ge=0)
    play_count_change: int
    mean_score_7d: float = Field(ge=0)
    best_score_7d: float = Field(ge=0)
    score_std_7d: float = Field(ge=0)

    # Required fields, but null is valid for single-play devices.
    mean_gap_hours: float | None = Field(ge=0, lt=168)
    max_gap_hours: float | None = Field(ge=0, lt=168)

    @field_validator("*", mode="before")
    @classmethod
    def require_numbers(cls, value):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Use JSON numbers, or null for missing gaps.")
        return value

    @model_validator(mode="after")
    def check_consistency(self):
        if self.active_days_7d > self.plays_7d:
            raise ValueError("Active days cannot exceed play count.")

        if self.plays_first_3d + self.plays_last_3d > self.plays_7d:
            raise ValueError("First/last three-day counts exceed total plays.")

        expected_change = self.plays_last_3d - self.plays_first_3d
        if self.play_count_change != expected_change:
            raise ValueError(
                "play_count_change must equal plays_last_3d - plays_first_3d."
            )

        if self.mean_score_7d > self.best_score_7d:
            raise ValueError("Mean score cannot exceed best score.")

        # A small tolerance accommodates rounded feature exports.
        if abs(
            self.days_since_last_play + self.activity_span_days - 7
        ) > 0.01:
            raise ValueError(
                "Recency plus activity span must approximately equal 7 days."
            )

        if self.plays_7d == 1:
            if self.mean_gap_hours is not None or self.max_gap_hours is not None:
                raise ValueError("Single-play gap features must be null.")

            if (
                self.active_days_7d != 1
                or self.plays_first_3d != 1
                or self.plays_last_3d != 0
                or self.activity_span_days != 0
                or self.days_since_last_play != 7
                or self.score_std_7d != 0
                or self.mean_score_7d != self.best_score_7d
            ):
                raise ValueError("Inconsistent single-play features.")

        else:
            if self.mean_gap_hours is None or self.max_gap_hours is None:
                raise ValueError("Multiple plays require both gap features.")

            if self.mean_gap_hours > self.max_gap_hours:
                raise ValueError("Mean gap cannot exceed maximum gap.")

        return self


@lru_cache(maxsize=1)
def load_frozen_model():
    # Verify bytes before deserializing our own trusted artifact.
    model_bytes = (RELEASE / "model.joblib").read_bytes()
    policy_bytes = (RELEASE / "decision_policy.json").read_bytes()

    with METRICS_PATH.open("r", encoding="utf-8-sig") as handle:
        metrics = json.load(handle)

    if hashlib.sha256(model_bytes).hexdigest() != metrics["model_sha256"]:
        raise ValueError("Frozen model hash mismatch.")

    if hashlib.sha256(policy_bytes).hexdigest() != metrics["policy_sha256"]:
        raise ValueError("Frozen policy hash mismatch.")

    policy = json.loads(policy_bytes.decode("utf-8-sig"))
    features = policy["features"]
    threshold = float(policy["threshold"])

    if (
        len(features) != len(GameplayFeatures.model_fields)
        or set(features) != set(GameplayFeatures.model_fields)
    ):
        raise ValueError("Request schema and model feature names differ.")

    if not 0 < threshold < 1:
        raise ValueError("Invalid frozen threshold.")

    if threshold != float(metrics["threshold"]):
        raise ValueError("Threshold differs from final evaluation.")

    model = joblib.load(io.BytesIO(model_bytes))

    if list(model.feature_names_in_) != features:
        raise ValueError("Model feature order differs from frozen policy.")

    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Unexpected model classes.")

    return model, features, threshold, classes.index(1)


@router.post("/predict")
def predict(features: GameplayFeatures):
    """Predict inactivity and return engagement-based analyst suggestions."""
    try:
        model, feature_order, threshold, positive_index = load_frozen_model()

        frame = pd.DataFrame(
            [features.model_dump()],
            columns=feature_order,
        ).astype(float)

        probability = float(
            model.predict_proba(frame)[0, positive_index]
        )

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")

        recommendation = engagement_details(
            frame=frame,
            probability=probability,
            threshold=threshold,
        )

    except Exception as error:
        logger.error("Inference failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Prediction service could not complete this request.",
        ) from error

    return {
        "model_version": "v1",
        "feature_version": "features_v1",
        "churn_probability": probability,
        "decision_threshold": threshold,
        "at_or_above_threshold": probability >= threshold,
        "rules_version": recommendation["rules_version"],
        "recommendation": recommendation,
        "observation_window_days": 7,
        "outcome_window_days": 14,
        "persisted": False,
        "interpretation": (
            "Estimated probability of no gameplay in the following 14 days. "
            "Below the threshold does not necessarily mean low risk. "
            "Segments describe early engagement. Retention suggestions are "
            "analyst-defined ideas, not proven treatments."
        ),
    }