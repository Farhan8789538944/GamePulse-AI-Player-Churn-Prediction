from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "plays_7d",
    "active_days_7d",
    "days_since_last_play",
    "activity_span_days",
    "plays_first_3d",
    "plays_last_3d",
    "play_count_change",
    "mean_score_7d",
    "best_score_7d",
    "score_std_7d",
    "mean_gap_hours",
    "max_gap_hours",
]


def main():
    events = pd.read_parquet(
        DATA_DIR / "game1_clean_events.parquet"
    )
    labels = pd.read_csv(
        DATA_DIR / "game1_labels.csv",
        dtype={"device": "string"},
    )

    for column in ["first_event", "cutoff", "outcome_end"]:
        labels[column] = pd.to_datetime(labels[column], utc=True)

    events["event_time"] = pd.to_datetime(
        events["event_time"], utc=True
    )

    joined = events.merge(
        labels[["device", "first_event", "cutoff"]],
        on="device",
        how="inner",
        validate="many_to_one",
    )

    # Only past events are allowed to influence features.
    observed = joined.loc[
        (joined["event_time"] >= joined["first_event"])
        & (joined["event_time"] < joined["cutoff"])
    ].copy()

    observed = observed.sort_values(["device", "event_time"])

    observed["relative_day"] = (
        (observed["event_time"] - observed["first_event"])
        .dt.total_seconds()
        .floordiv(86400)
        .astype(int)
    )

    observed["gap_hours"] = (
        observed.groupby("device")["event_time"]
        .diff()
        .dt.total_seconds()
        / 3600
    )

    grouped = observed.groupby("device")

    features = grouped.agg(
        plays_7d=("event_time", "size"),
        active_days_7d=("relative_day", "nunique"),
        last_observed_event=("event_time", "max"),
        mean_score_7d=("score", "mean"),
        best_score_7d=("score", "max"),
        score_std_7d=("score", "std"),
        mean_gap_hours=("gap_hours", "mean"),
        max_gap_hours=("gap_hours", "max"),
    )

    first_counts = (
        observed.loc[observed["relative_day"] < 3]
        .groupby("device")
        .size()
    )
    last_counts = (
        observed.loc[observed["relative_day"] >= 4]
        .groupby("device")
        .size()
    )

    features["plays_first_3d"] = first_counts.reindex(
        features.index, fill_value=0
    )
    features["plays_last_3d"] = last_counts.reindex(
        features.index, fill_value=0
    )
    features["play_count_change"] = (
        features["plays_last_3d"] - features["plays_first_3d"]
    )

    # One observed score has zero within-window variation by convention.
    features["score_std_7d"] = features["score_std_7d"].fillna(0)

    result = labels[
        ["device", "first_event", "cutoff", "outcome_end", "churn"]
    ].merge(
        features,
        left_on="device",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    result["days_since_last_play"] = (
        (result["cutoff"] - result["last_observed_event"])
        .dt.total_seconds() / 86400
    )
    result["activity_span_days"] = (
        (result["last_observed_event"] - result["first_event"])
        .dt.total_seconds() / 86400
    )

    # Verify observation counts against the earlier label audit.
    expected = labels.set_index("device")["observation_event_count"]
    actual = result.set_index("device")["plays_7d"]

    assert actual.eq(expected.reindex(actual.index)).all()
    assert result["device"].is_unique
    assert result["active_days_7d"].between(1, 7).all()
    assert result["last_observed_event"].lt(result["cutoff"]).all()
    assert observed["relative_day"].between(0, 6).all()

    metadata = [
        "device", "first_event", "cutoff", "outcome_end", "churn"
    ]
    result = result[metadata + FEATURE_COLUMNS]
    result = result.sort_values(["cutoff", "device"])

    output = DATA_DIR / "game1_features.parquet"
    result.to_parquet(output, index=False)

    print(f"Devices: {len(result):,}")
    print(f"Model features: {len(FEATURE_COLUMNS)}")
    print("\nMISSING FEATURE VALUES")
    print(result[FEATURE_COLUMNS].isna().sum().to_string())
    print(f"\nSaved to: {output}")
    print("\nMetadata and churn are NOT model input features.")


if __name__ == "__main__":
    main()