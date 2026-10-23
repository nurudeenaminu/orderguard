from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "rolling_threshold_oof_predictions.csv"

REPORT_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = REPORT_DIR / "rolling_calibration_summary.csv"
BINS_PATH = REPORT_DIR / "rolling_calibration_bins.csv"

N_BINS = 10


def evaluate_fold(group):
    y = group["actual_is_late"].astype(int).to_numpy()
    p = group["predicted_probability"].to_numpy()

    p = np.clip(p, 1e-15, 1 - 1e-15)

    bin_ids = pd.qcut(
        p,
        q=N_BINS,
        labels=False,
        duplicates="drop",
    )

    calibration_data = pd.DataFrame(
        {
            "bin": bin_ids,
            "actual": y,
            "probability": p,
        }
    ).dropna(subset=["bin"])

    bins = (
        calibration_data.groupby("bin", observed=True)
        .agg(
            orders=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_late_rate=("actual", "mean"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
        )
        .reset_index()
    )

    # Expected calibration error: weighted absolute difference
    # between average predictions and observed rates within bins.
    ece = (
        (bins["orders"] / bins["orders"].sum())
        * (bins["mean_predicted_probability"] - bins["observed_late_rate"]).abs()
    ).sum()

    summary = {
        "fold": group["fold"].iloc[0],
        "orders": len(group),
        "observed_late_rate": y.mean(),
        "mean_predicted_probability": p.mean(),
        "mean_probability_minus_observed_rate": p.mean() - y.mean(),
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "average_precision": average_precision_score(y, p),
        "expected_calibration_error": ece,
        "calibration_bins": len(bins),
    }

    bins.insert(0, "fold", group["fold"].iloc[0])

    return summary, bins


def main():
    print("Loading out-of-fold predictions...")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_PATH}")

    data = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "fold",
        "actual_is_late",
        "predicted_probability",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data[list(required)].isna().any().any():
        raise ValueError("Required prediction data contains missing values.")

    if not data["predicted_probability"].between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    summaries = []
    all_bins = []

    for fold, group in data.groupby("fold", sort=True):
        summary, bins = evaluate_fold(group)
        summaries.append(summary)
        all_bins.append(bins)

    summary_df = pd.DataFrame(summaries)
    bins_df = pd.concat(all_bins, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    bins_df.to_csv(BINS_PATH, index=False)

    print("\nCALIBRATION METRICS BY FOLD")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nCALIBRATION BINS")
    print(
        bins_df[
            [
                "fold",
                "bin",
                "orders",
                "mean_predicted_probability",
                "observed_late_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Calibration bins saved to: {BINS_PATH}")


if __name__ == "__main__":
    main()
