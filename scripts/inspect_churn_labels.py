from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "rawdata_game1.csv"
LABELS_PATH = ROOT / "data" / "processed" / "game1_labels.csv"
REPORT_PATH = ROOT / "reports" / "game1_labels_summary.txt"

OBSERVATION_DAYS = 7
PREDICTION_DAYS = 14


def main():
    df = pd.read_csv(DATA_PATH, dtype={"device": "string"})

    required = {"device", "score", "time"}
    if set(df.columns) != required:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    df["device"] = df["device"].str.strip().replace("", pd.NA)
    df["event_time"] = pd.to_datetime(
        df["time"], unit="s", utc=True, errors="raise"
    )

    # Provisional recording endpoint; not a player's last activity.
    coverage_end = df["event_time"].max()

    missing_ids = int(df["device"].isna().sum())
    df = df.dropna(subset=["device"]).copy()

    # Provisional deduplication rule: identical device, score, and time.
    # Without event IDs, we cannot prove these are duplicate events.
    duplicate_rows = int(
        df.duplicated(subset=["device", "score", "time"]).sum()
    )
    df = df.drop_duplicates(
        subset=["device", "score", "time"]
    ).copy()

    players = (
        df.groupby("device")["event_time"]
        .min()
        .rename("first_event")
        .to_frame()
    )

    players["cutoff"] = (
        players["first_event"]
        + pd.Timedelta(days=OBSERVATION_DAYS)
    )
    players["outcome_end"] = (
        players["cutoff"]
        + pd.Timedelta(days=PREDICTION_DAYS)
    )

    # Require complete follow-up under our coverage assumption.
    labels = players.loc[
        players["outcome_end"] <= coverage_end
    ].copy()

    if labels.empty:
        raise ValueError("No devices have enough follow-up coverage.")

    events = df.merge(
        labels[["cutoff", "outcome_end"]],
        left_on="device",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )

    # Half-open intervals avoid counting boundary events twice.
    in_observation = events["event_time"] < events["cutoff"]

    in_outcome = (
        (events["event_time"] >= events["cutoff"])
        & (events["event_time"] < events["outcome_end"])
    )

    observation_counts = (
        events.loc[in_observation].groupby("device").size()
    )
    outcome_counts = (
        events.loc[in_outcome].groupby("device").size()
    )

    labels["observation_event_count"] = (
        observation_counts.reindex(labels.index, fill_value=0)
        .astype("int64")
    )
    labels["outcome_event_count"] = (
        outcome_counts.reindex(labels.index, fill_value=0)
        .astype("int64")
    )
    labels["churn"] = (
        labels["outcome_event_count"] == 0
    ).astype("int8")

    # Basic checks on the generated labels.
    assert labels.index.is_unique
    assert labels["observation_event_count"].ge(1).all()
    assert labels["outcome_end"].le(coverage_end).all()
    assert labels["churn"].isin([0, 1]).all()
    assert not (in_observation & in_outcome).any()

    labels = labels.sort_values("first_event")

    counts = (
        labels["churn"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    monthly = (
        labels.assign(
            cohort_month=labels["first_event"].dt.strftime("%Y-%m")
        )
        .groupby("cohort_month")
        .agg(
            devices=("churn", "size"),
            churn_rate=("churn", "mean"),
        )
    )
    monthly["churn_percent"] = (
        monthly.pop("churn_rate") * 100
    ).round(2)

    lines = [
        "GAME 1 — INITIAL CHURN LABEL AUDIT",
        "",
        f"Observation window: {OBSERVATION_DAYS} days",
        f"Outcome window: {PREDICTION_DAYS} days",
        f"Assumed coverage endpoint: {coverage_end}",
        "",
        f"Missing-ID records excluded: {missing_ids:,}",
        f"Duplicate records excluded: {duplicate_rows:,}",
        f"Remaining gameplay records: {len(df):,}",
        f"Eligible devices: {len(labels):,}",
        (
            "Devices excluded for incomplete follow-up: "
            f"{len(players) - len(labels):,}"
        ),
        "",
        "CLASS DISTRIBUTION",
    ]

    for label, description in [
        (0, "Returned during outcome window"),
        (1, "Inactive throughout outcome window"),
    ]:
        count = int(counts[label])
        percentage = 100 * count / len(labels)
        lines.append(
            f"churn={label}: {count:,} ({percentage:.2f}%)"
            f" — {description}"
        )

    lines.extend([
        "",
        "CHURN BY FIRST-OBSERVED MONTH",
        monthly.to_string(),
        "",
        "LIMITATIONS",
        "- A device is a player proxy, not a verified individual.",
        "- First observed activity is not necessarily registration.",
        "- Complete recording through the endpoint is assumed.",
        "- Inactivity during the outcome window is not permanent churn.",
        "- Duplicate handling is a documented provisional rule.",
        "",
        "LEAKAGE WARNING",
        "outcome_event_count and outcome_end are audit/label fields.",
        "Do not use them as model features.",
    ])

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels.reset_index().to_csv(LABELS_PATH, index=False)

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nLabels saved to: {LABELS_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()