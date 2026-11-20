import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import joblib

from gamepulse.retention import RULE_CONFIG, recommend


ROOT = Path(__file__).resolve().parents[2]
SEGMENT_DIR = ROOT / "artifacts/segmentation"

SEGMENT_VERSION = "segments_v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=1)
def load_segmentation():
    model_bytes = (SEGMENT_DIR / "kmeans_pipeline.joblib").read_bytes()
    labels = read_json(SEGMENT_DIR / "segment_labels.json")
    metadata = read_json(SEGMENT_DIR / "metadata.json")

    actual_hash = hashlib.sha256(model_bytes).hexdigest()
    if actual_hash != labels["segmentation_sha256"]:
        raise ValueError("Segmentation artifact hash mismatch.")

    saved_rules = read_json(
        ROOT / "reports/recommendations/rules_config.json"
    )
    if saved_rules != RULE_CONFIG:
        raise ValueError("Shared rules differ from the saved rule configuration.")

    if metadata["selected_k"] != 2:
        raise ValueError("Expected the reviewed two-cluster model.")

    names = labels["names"]
    if set(names) != {"0", "1"}:
        raise ValueError("Expected segment names for IDs 0 and 1.")

    model = joblib.load(io.BytesIO(model_bytes))
    features = metadata["features"]

    if list(model.feature_names_in_) != features:
        raise ValueError("Segmentation feature order mismatch.")

    return model, features, names


def engagement_details(frame, probability, threshold):
    model, feature_names, names = load_segmentation()

    segment_id = int(model.predict(frame[feature_names])[0])
    segment_name = names.get(str(segment_id))

    if segment_name is None:
        raise ValueError("Unexpected segment ID.")

    row = frame.iloc[0]

    suggestion = recommend(
        plays_7d=int(row["plays_7d"]),
        days_since_last_play=float(row["days_since_last_play"]),
        churn_probability=probability,
        threshold=threshold,
        config=RULE_CONFIG,
    )

    # Match the structure already stored by batch scoring.
    suggestion.update(
        {
            "rules_version": RULE_CONFIG["version"],
            "rules_config": dict(RULE_CONFIG),
            "segment_version": SEGMENT_VERSION,
            "segment_id": segment_id,
            "segment_name": segment_name,
        }
    )

    return suggestion