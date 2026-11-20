import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, HTTPException
from scipy.special import expit

from gamepulse.players import get_player
from gamepulse.predict import load_frozen_model


ROOT = Path(__file__).resolve().parents[2]
EXPLAINER_DIR = ROOT / "artifacts/explanations/v1"
METRICS_PATH = ROOT / "reports/final_test/metrics.json"

router = APIRouter(prefix="/v1/players", tags=["Explanations"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_explainer():
    pipeline, features, threshold, positive_index = load_frozen_model()

    # Binary Logistic Regression decision_function explains class 1 here.
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected class order [0, 1].")

    metadata = json.loads(
        (EXPLAINER_DIR / "metadata.json").read_text(
            encoding="utf-8-sig"
        )
    )
    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8-sig")
    )

    if metadata["release"] != "v1":
        raise ValueError("Unexpected explanation release.")

    if metadata["model_sha256"] != metrics["model_sha256"]:
        raise ValueError("Explanation metadata references a different model.")

    if metadata["background_partition"] != "train":
        raise ValueError("Explanation background must come from training data.")

    if metadata["explanation_units"] != "log_odds_of_inactivity":
        raise ValueError("Unexpected explanation units.")

    background = pd.read_parquet(
        EXPLAINER_DIR / "background_raw.parquet"
    )

    if list(background.columns) != features:
        raise ValueError("Background feature order differs from the model.")

    if len(background) != metadata["background_size"]:
        raise ValueError("Background size differs from metadata.")

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["model"]

    transformed_background = scaler.transform(
        imputer.transform(background[features])
    )

    saved_background = np.load(
        EXPLAINER_DIR / "background_transformed.npy",
        allow_pickle=False,
    )

    if not np.isfinite(transformed_background).all():
        raise ValueError("Background contains invalid transformed values.")

    np.testing.assert_allclose(
        transformed_background,
        saved_background,
        rtol=1e-6,
        atol=1e-6,
    )

    names = imputer.get_feature_names_out(features).tolist()

    if names != metadata["transformed_feature_names"]:
        raise ValueError("Transformed feature names differ from metadata.")

    masker = shap.maskers.Independent(
        transformed_background,
        max_samples=len(transformed_background),
    )

    explainer = shap.LinearExplainer(
        classifier,
        masker,
        feature_names=names,
    )

    return (
        pipeline,
        features,
        positive_index,
        imputer,
        scaler,
        names,
        explainer,
        len(background),
    )


@router.get("/{device_id}/explanation")
def player_explanation(device_id: str):
    """Explain a stored validation prediction using the fixed background."""

    # Reuse the existing validated database retrieval.
    # Unknown devices retain its HTTP 404 response.
    player = get_player(device_id)

    try:
        (
            pipeline,
            features,
            positive_index,
            imputer,
            scaler,
            names,
            explainer,
            background_size,
        ) = load_explainer()

        frame = pd.DataFrame(
            [player["features"]],
            columns=features,
        ).astype(float)

        transformed = scaler.transform(imputer.transform(frame))
        explanation = explainer(transformed)

        contributions = np.asarray(
            explanation.values, dtype=float
        ).reshape(-1)

        base_log_odds = float(
            np.asarray(explanation.base_values).reshape(-1)[0]
        )

        if len(contributions) != len(names):
            raise ValueError("Unexpected number of SHAP contributions.")

        if (
            not np.isfinite(contributions).all()
            or not np.isfinite(base_log_odds)
        ):
            raise ValueError("Explanation contains non-finite values.")

        reconstructed_log_odds = (
            base_log_odds + float(contributions.sum())
        )
        reconstructed_probability = float(
            expit(reconstructed_log_odds)
        )

        model_log_odds = float(
            pipeline.decision_function(frame)[0]
        )
        model_probability = float(
            pipeline.predict_proba(frame)[0, positive_index]
        )

        np.testing.assert_allclose(
            reconstructed_log_odds,
            model_log_odds,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            reconstructed_probability,
            model_probability,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            model_probability,
            player["churn_probability"],
            rtol=0,
            atol=1e-12,
        )

        items = []
        for name, value in zip(names, contributions):
            value = float(value)
            items.append(
                {
                    "feature": name,
                    "shap_log_odds": value,
                    "direction": (
                        "increases predicted inactivity"
                        if value > 0
                        else (
                            "decreases predicted inactivity"
                            if value < 0
                            else "no contribution"
                        )
                    ),
                }
            )

        items.sort(
            key=lambda item: abs(item["shap_log_odds"]),
            reverse=True,
        )

        return {
            "device_id": player["device_id"],
            "snapshot_id": player["snapshot_id"],
            "partition": player["data_partition"],
            "model_version": player["model_version"],
            "churn_probability": model_probability,
            "decision_threshold": player["decision_threshold"],
            "at_or_above_threshold": player["at_or_above_threshold"],
            "method": "SHAP LinearExplainer with independent masker",
            "units": "log_odds_of_inactivity",
            "background_partition": "train",
            "background_devices": background_size,
            "base_log_odds": base_log_odds,
            "contribution_sum": float(contributions.sum()),
            "reconstructed_probability": reconstructed_probability,
            "reconstruction_verified": True,
            "contributions": items,
            "limitations": [
                "Contributions explain model predictions, not causes.",
                "Independent-feature attribution does not model correlations.",
                "Several features are related or redundant.",
                "Both missing-gap indicators describe single-play missingness.",
                "Contributions are in log-odds, not percentage points.",
                "The reference is a fixed sample of training devices.",
            ],
        }

    except Exception as error:
        logger.error(
            "Player explanation failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="A verified player explanation is unavailable.",
        ) from error