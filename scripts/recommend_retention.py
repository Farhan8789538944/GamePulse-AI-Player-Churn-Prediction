# Allow this standalone script to use the shared application package.
import sys
from pathlib import Path as _Path

_src = str(_Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from gamepulse.retention import (
    SEGMENT_NAMES,
    RULE_CONFIG,
    recommend,
    check_rule_boundaries,
)

from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RELEASE_DIR = ROOT / "artifacts" / "releases" / "v1"
SEGMENT_DIR = ROOT / "artifacts" / "segmentation"
REPORT_DIR = ROOT / "reports" / "recommendations"


# Analyst-defined rules, not learned or validated interventions.


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()






def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = RELEASE_DIR / "model.joblib"
    policy_path = RELEASE_DIR / "decision_policy.json"
    segment_path = SEGMENT_DIR / "kmeans_pipeline.joblib"

    final_results = json.loads(
        (ROOT / "reports" / "final_test" / "metrics.json")
        .read_text(encoding="utf-8")
    )

    if sha256(model_path) != final_results["model_sha256"]:
        raise ValueError("Frozen churn model hash mismatch.")

    if sha256(policy_path) != final_results["policy_sha256"]:
        raise ValueError("Frozen decision policy hash mismatch.")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    segment_metadata = json.loads(
        (SEGMENT_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    if segment_metadata["selected_k"] != 2:
        raise ValueError("Segment names require the reviewed two-cluster model.")

    threshold = float(policy["threshold"])
    check_rule_boundaries(threshold)

    model = joblib.load(model_path)
    segment_model = joblib.load(segment_path)

    validation = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    )

    churn_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        validation[policy["features"]]
    )[:, churn_index]

    segment_ids = segment_model.predict(
        validation[segment_metadata["features"]]
    )

    if not set(np.unique(segment_ids)).issubset(SEGMENT_NAMES):
        raise ValueError("Unexpected segment IDs.")

    records = []

    # Deliberately exclude actual outcomes from recommendation inputs.
    for position, (_, row) in enumerate(validation.iterrows()):
        probability = float(probabilities[position])
        segment_id = int(segment_ids[position])

        suggestion = recommend(
            plays_7d=int(row["plays_7d"]),
            days_since_last_play=float(row["days_since_last_play"]),
            churn_probability=probability,
            threshold=threshold,
            config=RULE_CONFIG,
        )

        records.append({
            "device": str(row["device"]),
            "prediction_cutoff": row["cutoff"].isoformat(),
            "churn_probability": probability,
            "segment_id": segment_id,
            "segment_name": SEGMENT_NAMES[segment_id],
            "plays_7d": int(row["plays_7d"]),
            "days_since_last_play": float(row["days_since_last_play"]),
            **suggestion,
            "model_version": "v1",
            "rules_version": RULE_CONFIG["version"],
        })

    output = pd.DataFrame(records)

    assert len(output) == len(validation)
    assert output["device"].is_unique
    assert output["rule_id"].notna().all()

    output.to_csv(
        REPORT_DIR / "validation_recommendations.csv",
        index=False,
    )

    # Store labels tied to this exact segmentation artifact.
    segment_labels = {
        "segmentation_sha256": sha256(segment_path),
        "names": {
            str(key): value for key, value in SEGMENT_NAMES.items()
        },
        "basis": "Reviewed training behavioural profiles",
    }

    (SEGMENT_DIR / "segment_labels.json").write_text(
        json.dumps(segment_labels, indent=2),
        encoding="utf-8",
    )

    (REPORT_DIR / "rules_config.json").write_text(
        json.dumps(RULE_CONFIG, indent=2),
        encoding="utf-8",
    )

    print("Frozen churn model and policy hashes verified.")
    print("Rule boundary and invalid-input checks passed.")
    print(f"Devices processed: {len(output):,}")
    print("Actual future outcomes were not used for recommendations.")

    print("\nRECOMMENDATION COUNTS")
    print(output["rule_id"].value_counts().to_string())

    print("\nSEGMENT COUNTS")
    print(output["segment_name"].value_counts().to_string())

    print("\nEXAMPLE RECOMMENDATION")
    print(json.dumps(records[0], indent=2))

    print(f"\nReports saved to: {REPORT_DIR}")
    print("These are analyst suggestions, not proven retention treatments.")
    print("No messages were sent and no player actions were taken.")


if __name__ == "__main__":
    main()