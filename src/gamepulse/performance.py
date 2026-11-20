import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports/final_test/metrics.json"
RELEASE_DIR = ROOT / "artifacts/releases/v1"

router = APIRouter(prefix="/v1", tags=["Model performance"])
logger = logging.getLogger(__name__)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_performance():
    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8-sig")
    )

    # Reject non-finite JSON values before returning a response.
    json.dumps(metrics, allow_nan=False)

    if metrics["release"] != "v1":
        raise ValueError("Unexpected model release.")

    if sha256(RELEASE_DIR / "model.joblib") != metrics["model_sha256"]:
        raise ValueError("Model does not match the evaluated artifact.")

    policy_path = RELEASE_DIR / "decision_policy.json"

    if sha256(policy_path) != metrics["policy_sha256"]:
        raise ValueError("Policy does not match the evaluated artifact.")

    policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))

    if float(policy["threshold"]) != float(metrics["threshold"]):
        raise ValueError("Evaluation and policy thresholds differ.")

    matrix = metrics["confusion_matrix"]

    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        raise ValueError("Expected a 2-by-2 confusion matrix.")

    if any(
        type(value) is not int or value < 0
        for row in matrix
        for value in row
    ):
        raise ValueError("Confusion matrix counts must be nonnegative integers.")

    total = sum(sum(row) for row in matrix)

    if total != metrics["test_devices"] or total <= 0:
        raise ValueError("Confusion matrix and test count disagree.")

    report = metrics["classification_report"]

    if (
        sum(matrix[0]) != report["returned"]["support"]
        or sum(matrix[1]) != report["inactive"]["support"]
    ):
        raise ValueError("Confusion matrix and class supports disagree.")

    return {
        "model_version": metrics["release"],
        "model_name": "Logistic Regression",
        "evaluation_partition": "test",
        "evaluation_type": "saved_final_evaluation",
        "decision_threshold": metrics["threshold"],
        "test_devices": metrics["test_devices"],
        "observed_inactivity_rate": metrics["test_churn_prevalence"],
        "model_metrics": metrics["model_metrics"],
        "classification_report": report,
        "confusion_matrix": {
            "class_order": ["returned", "inactive"],
            "rows": "actual",
            "columns": "predicted",
            "counts": matrix,
        },
        "baselines": {
            "training_prior_dummy": metrics["dummy_metrics"],
            "recency_ranking": metrics["recency_ranking_metrics"],
        },
        "top_risk_ranking": metrics["top_risk_ranking"],
        "limitations": metrics["limitations"],
        "notes": [
            "These results are from the held-out chronological test partition.",
            "The player browser displays validation devices, a different cohort.",
            "Recency is a ranking baseline; missing metrics are not zero.",
            "Top-risk results account for ties at the selection boundary.",
            "High inactivity prevalence makes accuracy alone misleading.",
            "The model does not outperform recency on every ranking metric.",
        ],
        "artifact_hashes_verified": True,
    }


@router.get("/model-performance")
def model_performance():
    """Return saved final-test results for the frozen v1 release."""
    try:
        return load_performance()

    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.error(
            "Performance report unavailable: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Verified model-performance results are unavailable.",
        ) from error