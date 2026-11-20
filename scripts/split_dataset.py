from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "split_summary.txt"


def main():
    df = pd.read_parquet(DATA_DIR / "game1_features.parquet")
    df = df.sort_values(["cutoff", "device"]).reset_index(drop=True)

    # Boundaries depend only on prediction times, not target values.
    validation_start = df.loc[int(len(df) * 0.60), "cutoff"]
    test_start = df.loc[int(len(df) * 0.80), "cutoff"]

    if validation_start >= test_start:
        raise ValueError("Insufficient distinct times for these splits.")

    train = df.loc[
        (df["cutoff"] < validation_start)
        & (df["outcome_end"] <= validation_start)
    ].copy()

    validation = df.loc[
        (df["cutoff"] >= validation_start)
        & (df["cutoff"] < test_start)
        & (df["outcome_end"] <= test_start)
    ].copy()

    test = df.loc[df["cutoff"] >= test_start].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, part in splits.items():
        if part.empty:
            raise ValueError(f"{name} partition is empty.")
        if part["churn"].nunique() != 2:
            raise ValueError(f"{name} does not contain both classes.")

    assert set(train["device"]).isdisjoint(validation["device"])
    assert set(train["device"]).isdisjoint(test["device"])
    assert set(validation["device"]).isdisjoint(test["device"])
    assert train["outcome_end"].max() <= validation["cutoff"].min()
    assert validation["outcome_end"].max() <= test["cutoff"].min()

    excluded = len(df) - sum(len(part) for part in splits.values())

    lines = [
        "CHRONOLOGICAL SPLIT SUMMARY",
        "",
        f"Validation starts: {validation_start}",
        f"Test starts: {test_start}",
        f"Examples excluded for boundary label maturity: {excluded:,}",
    ]

    for name, part in splits.items():
        output = DATA_DIR / f"game1_{name}.parquet"
        part.to_parquet(output, index=False)

        counts = part["churn"].value_counts()

        lines.extend([
            "",
            name.upper(),
            f"Devices: {len(part):,}",
            f"Returning devices: {int(counts.get(0, 0)):,}",
            f"Inactive devices: {int(counts.get(1, 0)):,}",
            f"Prediction times: {part['cutoff'].min()}",
            f"              to: {part['cutoff'].max()}",
        ])

    lines.extend([
        "",
        "No scaling, imputation, or resampling has been fitted.",
        "Earlier descriptive EDA included the full dataset.",
        "Future model selection must use train/validation only.",
        "Reserve the test partition for final evaluation.",
    ])

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()