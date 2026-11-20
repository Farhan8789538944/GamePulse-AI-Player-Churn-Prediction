from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RELEASE_DIR = ROOT / "artifacts" / "releases" / "v1"
REPORT_DIR = ROOT / "reports" / "subgroups"

# Reporting heuristic, not a statistical guarantee.
MIN_DEVICES = 100
MIN_PER_CLASS = 30


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def summarize_group(name, group):
    y = group["churn"].astype(int).to_numpy()
    probabilities = group["churn_probability"].to_numpy()
    predictions = group["predicted_churn"].to_numpy()

    matrix = confusion_matrix(
        y, predictions, labels=[0, 1]
    )

    returned_correct = int(matrix[0, 0])
    returned_as_inactive = int(matrix[0, 1])
    inactive_as_returned = int(matrix[1, 0])
    inactive_correct = int(matrix[1, 1])

    returned_count = int((y == 0).sum())
    inactive_count = int((y == 1).sum())

    return_recall = safe_divide(
        returned_correct, returned_count
    )
    churn_recall = safe_divide(
        inactive_correct, inactive_count
    )

    has_both_classes = returned_count > 0 and inactive_count > 0

    limited_support = (
        len(group) < MIN_DEVICES
        or min(returned_count, inactive_count) < MIN_PER_CLASS
    )

    return {
        "group": str(name),
        "devices": len(group),
        "returned_devices": returned_count,
        "inactive_devices": inactive_count,
        "observed_churn": float(y.mean()),
        "mean_predicted_churn": float(probabilities.mean()),
        "mean_prediction_minus_observed": float(
            probabilities.mean() - y.mean()
        ),
        "return_recall": return_recall,
        "return_precision": safe_divide(
            returned_correct,
            returned_correct + inactive_as_returned,
        ),
        "churn_recall": churn_recall,
        "churn_precision": safe_divide(
            inactive_correct,
            inactive_correct + returned_as_inactive,
        ),
        "balanced_accuracy": (
            (return_recall + churn_recall) / 2
            if has_both_classes else np.nan
        ),
        "brier_score": float(
            brier_score_loss(y, probabilities)
        ),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if has_both_classes else np.nan
        ),
        "return_average_precision": (
            float(
                average_precision_score(
                    1 - y, 1 - probabilities
                )
            )
            if has_both_classes else np.nan
        ),
        "limited_support": limited_support,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    policy_path = RELEASE_DIR / "decision_policy.json"
    model_path = RELEASE_DIR / "model.joblib"

    final_results = json.loads(
        (ROOT / "reports" / "final_test" / "metrics.json")
        .read_text(encoding="utf-8")
    )

    if sha256(model_path) != final_results["model_sha256"]:
        raise ValueError("Frozen model hash mismatch.")

    if sha256(policy_path) != final_results["policy_sha256"]:
        raise ValueError("Frozen policy hash mismatch.")

    policy = json.loads(
        policy_path.read_text(encoding="utf-8")
    )
    model = joblib.load(model_path)

    # Validation diagnostics only.
    # We do not reopen test data or change the frozen model.
    data = pd.read_parquet(
        DATA_DIR / "game1_validation.parquet"
    ).copy()

    churn_index = list(model.classes_).index(1)

    data["churn_probability"] = model.predict_proba(
        data[policy["features"]]
    )[:, churn_index]

    data["predicted_churn"] = (
        data["churn_probability"] >= policy["threshold"]
    ).astype(int)

    data["activity_group"] = pd.cut(
        data["plays_7d"],
        bins=[0, 1, 4, 9, np.inf],
        labels=["1 play", "2-4 plays", "5-9 plays", "10+ plays"],
    )

    data["recency_group"] = pd.cut(
        data["days_since_last_play"],
        bins=[-np.inf, 2, 5, np.inf],
        labels=["0-2 days", ">2-5 days", ">5 days"],
    )

    data["prediction_month"] = pd.to_datetime(
        data["cutoff"], utc=True
    ).dt.strftime("%Y-%m")

    group_columns = [
        "activity_group",
        "recency_group",
        "prediction_month",
    ]

    display_columns = [
        "group",
        "devices",
        "returned_devices",
        "observed_churn",
        "return_recall",
        "churn_recall",
        "balanced_accuracy",
        "brier_score",
        "limited_support",
    ]

    report_lines = [
        "FROZEN V1 — VALIDATION SUBGROUP DIAGNOSTICS",
        "",
        f"Threshold: {policy['threshold']}",
        f"Validation devices: {len(data):,}",
        "No model, threshold, or feature changes performed.",
    ]

    for column in group_columns:
        if data[column].isna().any():
            raise ValueError(f"Missing group assignments: {column}")

        rows = [
            summarize_group(name, group)
            for name, group in data.groupby(
                column, observed=True, sort=True
            )
        ]

        table = pd.DataFrame(rows)

        # Every device must occur exactly once within this grouping.
        assert int(table["devices"].sum()) == len(data)

        table.to_csv(
            REPORT_DIR / f"{column}.csv",
            index=False,
        )

        report_lines.extend([
            "",
            column.upper(),
            table[display_columns].round(4).to_string(index=False),
        ])

    report_lines.extend([
        "",
        "INTERPRETATION",
        "limited_support=True indicates small total or class counts.",
        "This flag is a heuristic, not a confidence interval.",
        "Group prevalence differs; precision and AP are not directly",
        "comparable without considering class proportions.",
        "Brier differences also reflect different group difficulty.",
        "Average prediction minus observed rate is only an aggregate",
        "calibration check, not a full calibration assessment.",
        "These are validation diagnostics, not independent test results.",
        "No demographic fairness or causal claims are supported.",
    ])

    report = "\n".join(report_lines)

    (REPORT_DIR / "summary.txt").write_text(
        report, encoding="utf-8"
    )

    print(report)
    print(f"\nDetailed reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()