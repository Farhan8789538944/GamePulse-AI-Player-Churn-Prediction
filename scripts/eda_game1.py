from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "rawdata_game1.csv"
LABELS_PATH = ROOT / "data" / "processed" / "game1_labels.csv"
REPORT_DIR = ROOT / "reports" / "eda"
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Apply the same documented cleaning rules as Module 2.
    events = pd.read_csv(RAW_PATH, dtype={"device": "string"})
    events["device"] = (
        events["device"].str.strip().replace("", pd.NA)
    )
    events = events.dropna(subset=["device"])
    events = events.drop_duplicates(
        subset=["device", "score", "time"]
    ).copy()

    events["event_time"] = pd.to_datetime(
        events["time"], unit="s", utc=True, errors="raise"
    )
    events = events.sort_values(["device", "event_time"])

    # Preserve cleaned data separately from the unchanged raw file.
    events.to_parquet(
        PROCESSED_DIR / "game1_clean_events.parquet",
        index=False,
    )

    labels = pd.read_csv(
        LABELS_PATH, dtype={"device": "string"}
    )
    for column in ["first_event", "cutoff", "outcome_end"]:
        labels[column] = pd.to_datetime(labels[column], utc=True)

    if labels["device"].duplicated().any():
        raise ValueError("Label file contains repeated devices.")

    # Bring in only the metadata needed for observation-window EDA.
    joined = events.merge(
        labels[["device", "first_event", "cutoff"]],
        on="device",
        how="inner",
        validate="many_to_one",
    )

    observed = joined.loc[
        (joined["event_time"] >= joined["first_event"])
        & (joined["event_time"] < joined["cutoff"])
    ].copy()

    # Relative 24-hour periods since first activity, indexed 0 to 6.
    observed["relative_day"] = (
        (observed["event_time"] - observed["first_event"])
        .dt.total_seconds()
        .floordiv(86400)
        .astype(int)
    )

    activity = observed.groupby("device").agg(
        plays_7d=("event_time", "size"),
        active_days_7d=("relative_day", "nunique"),
        last_observed_event=("event_time", "max"),
        mean_score_7d=("score", "mean"),
        best_score_7d=("score", "max"),
    )

    summary = labels[
        ["device", "first_event", "cutoff", "churn"]
    ].merge(
        activity,
        left_on="device",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    summary["days_since_last_play"] = (
        (summary["cutoff"] - summary["last_observed_event"])
        .dt.total_seconds() / 86400
    )

    # Check that this independent aggregation matches Module 2.
    expected_counts = labels.set_index("device")[
        "observation_event_count"
    ]
    actual_counts = summary.set_index("device")["plays_7d"]
    assert actual_counts.eq(
        expected_counts.reindex(actual_counts.index)
    ).all()

    assert summary["active_days_7d"].between(1, 7).all()
    assert summary["days_since_last_play"].between(0, 7).all()

    summary["activity_group"] = pd.cut(
        summary["plays_7d"],
        bins=[0, 1, 4, 9, float("inf")],
        labels=["1 play", "2-4 plays", "5-9 plays", "10+ plays"],
    )

    activity_table = (
        summary.groupby("activity_group", observed=True)
        .agg(
            devices=("churn", "size"),
            churn_rate=("churn", "mean"),
        )
    )
    activity_table["churn_percent"] = (
        activity_table.pop("churn_rate") * 100
    ).round(2)

    feature_columns = [
        "plays_7d",
        "active_days_7d",
        "days_since_last_play",
        "mean_score_7d",
        "best_score_7d",
    ]
    medians = summary.groupby("churn")[feature_columns].median()

    # Diagnostic cohort only: do not change the target population yet.
    recently_active = summary.loc[
        summary["days_since_last_play"] <= 2
    ]

    counts = (
        summary["churn"].value_counts()
        .reindex([0, 1], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Returned", "Inactive"],
        counts.to_numpy(),
        color=["#2a9d8f", "#e76f51"],
    )
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Eligible devices")
    ax.set_title("Activity during the 14-day outcome window")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    activity_table["churn_percent"].plot.bar(
        ax=ax, color="#457b9d", rot=0
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel("Gameplay count in the observation window")
    ax.set_ylabel("Future inactivity (%)")
    ax.set_title("Observed activity and future inactivity")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "churn_by_activity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name in [(0, "Returned"), (1, "Inactive")]:
        values = summary.loc[
            summary["churn"] == label, "days_since_last_play"
        ]
        ax.hist(
            values,
            bins=14,
            range=(0, 7),
            density=True,
            alpha=0.5,
            label=name,
        )
    ax.set_xlabel("Days since last play at prediction time")
    ax.set_ylabel("Density within each class")
    ax.set_title("Recency before prediction, grouped by future outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "recency_by_outcome.png", dpi=150)
    plt.close(fig)

    lines = [
        "GAME 1 — INITIAL EXPLORATORY ANALYSIS",
        "",
        f"Clean event records: {len(events):,}",
        f"Eligible devices: {len(summary):,}",
        "",
        "FUTURE INACTIVITY BY OBSERVED PLAY COUNT",
        activity_table.to_string(),
        "",
        "MEDIAN OBSERVATION-WINDOW FEATURES BY OUTCOME",
        "0 = returned; 1 = inactive",
        medians.round(2).to_string(),
        "",
        "RECENTLY ACTIVE COHORT — DIAGNOSTIC ONLY",
        "Devices with gameplay in the last 48 hours before cutoff:",
        f"{len(recently_active):,}",
    ]

    if not recently_active.empty:
        lines.append(
            "Future inactivity in this cohort: "
            f"{recently_active['churn'].mean() * 100:.2f}%"
        )

    lines.extend([
        "",
        "These are descriptive associations, not causal effects.",
        "The recently active cohort has not replaced our population.",
        "This EDA summary is not a model-ready feature dataset.",
    ])

    report = "\n".join(lines)
    (REPORT_DIR / "summary.txt").write_text(
        report, encoding="utf-8"
    )
    activity_table.to_csv(REPORT_DIR / "activity_groups.csv")

    print(report)
    print(f"\nCharts and tables saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()