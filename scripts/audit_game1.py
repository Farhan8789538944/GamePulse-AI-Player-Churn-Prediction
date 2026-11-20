from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "rawdata_game1.csv"
REPORT_PATH = ROOT / "reports" / "game1_audit.txt"


def main():
    # Read identifiers as text to preserve their exact values.
    df = pd.read_csv(DATA_PATH, dtype={"device": "string"})

    expected_columns = {"device", "score", "time"}
    if set(df.columns) != expected_columns:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    df["device"] = df["device"].str.strip()
    df["device"] = df["device"].replace("", pd.NA)

    scores = pd.to_numeric(df["score"], errors="coerce")
    timestamps = pd.to_numeric(df["time"], errors="coerce")
    dates = pd.to_datetime(
        timestamps, unit="s", utc=True, errors="coerce"
    )

    lines = []

    def record(title, value):
        lines.append(f"\n{title}\n{value}")

    record("FILE", DATA_PATH.name)
    record("TOTAL GAMEPLAY RECORDS", f"{len(df):,}")
    record("UNIQUE DEVICE IDS", f"{df['device'].nunique():,}")
    record("COLUMN TYPES", df.dtypes.to_string())
    record("MISSING VALUES", df.isna().sum().to_string())

    record(
        "EXACT DUPLICATE ROWS (EXCLUDING FIRST OCCURRENCE)",
        f"{df.duplicated().sum():,}",
    )
    record("INVALID OR MISSING SCORES", int(scores.isna().sum()))
    record("NEGATIVE SCORES", int((scores < 0).sum()))
    record("INVALID OR MISSING TIMESTAMPS", int(dates.isna().sum()))
    record("SCORE SUMMARY", scores.describe().to_string())

    # Use valid identifiers and dates for the temporal audit.
    valid = pd.DataFrame({
        "device": df["device"],
        "event_time": dates,
    }).dropna(subset=["device", "event_time"])

    if valid.empty:
        record("TEMPORAL AUDIT", "No valid device/timestamp pairs.")
    else:
        first_time = valid["event_time"].min()
        last_time = valid["event_time"].max()

        record("FIRST EVENT (UTC)", first_time)
        record("LAST EVENT (UTC)", last_time)
        record(
            "OBSERVED DATE SPAN IN DAYS",
            round((last_time - first_time).total_seconds() / 86400, 2),
        )

        per_device = valid.groupby("device").agg(
            event_count=("event_time", "size"),
            first_event=("event_time", "min"),
            last_event=("event_time", "max"),
        )

        record(
            "GAMEPLAY RECORDS PER DEVICE",
            per_device["event_count"].describe().to_string(),
        )
        record(
            "DEVICES WITH ONLY ONE VALID RECORD",
            int((per_device["event_count"] == 1).sum()),
        )

        # Preliminary coverage check, not final label eligibility.
        # Assumes logging continued through the dataset's last event.
        window_results = []
        for observation_days, prediction_days in [(7, 14), (7, 30)]:
            required_days = observation_days + prediction_days
            candidate_count = (
                per_device["first_event"]
                + pd.Timedelta(days=required_days)
                <= last_time
            ).sum()

            window_results.append(
                f"{observation_days}-day observation + "
                f"{prediction_days}-day prediction: "
                f"{candidate_count:,} candidate devices"
            )

        record(
            "PRELIMINARY FOLLOW-UP COVERAGE",
            "\n".join(window_results),
        )

        daily_counts = (
            valid.set_index("event_time")
            .resample("D")
            .size()
        )
        record(
            "DAYS WITH ZERO EVENTS BETWEEN FIRST AND LAST EVENT",
            int((daily_counts == 0).sum()),
        )
        record(
            "MONTHLY VALID EVENT COUNTS",
            valid.set_index("event_time")
            .resample("MS")
            .size()
            .to_string(),
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()