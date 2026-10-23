from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

EXCLUDED_FEATURES = {
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
    "predicted_is_late",
    "predicted_late_probability",
    "prediction_outcome",
}


def main():
    print("Loading test data and saved predictions...")

    test = pd.read_csv(TEST_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    data = test.merge(
        predictions[
            [
                "order_id",
                "predicted_is_late",
                "predicted_late_probability",
                "prediction_outcome",
            ]
        ],
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    if data["prediction_outcome"].isna().any():
        raise ValueError("Some test orders do not have a saved prediction.")

    data["predicted_is_late"] = data["predicted_is_late"].astype(int)
    data["is_late"] = data["is_late"].astype(int)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. How many correct and incorrect predictions?
    outcome_summary = (
        data.groupby("prediction_outcome")
        .agg(
            orders=("order_id", "size"),
            actual_late_orders=("is_late", "sum"),
            predicted_late_orders=("predicted_is_late", "sum"),
            mean_predicted_probability=("predicted_late_probability", "mean"),
        )
        .reset_index()
    )

    outcome_summary["share_of_test_set"] = outcome_summary["orders"] / len(data)

    outcome_path = REPORT_DIR / "error_outcome_summary.csv"
    outcome_summary.to_csv(outcome_path, index=False)

    print("\nPREDICTION OUTCOMES")
    print(outcome_summary.to_string(index=False))

    # 2. Compare numeric features across error types.
    numeric_features = [
        col
        for col in data.select_dtypes(include=np.number).columns
        if col not in EXCLUDED_FEATURES
    ]

    numeric_rows = []

    for feature in numeric_features:
        for outcome, group in data.groupby("prediction_outcome"):
            values = group[feature].dropna()

            numeric_rows.append(
                {
                    "feature": feature,
                    "prediction_outcome": outcome,
                    "non_missing_count": len(values),
                    "median": values.median() if not values.empty else np.nan,
                    "mean": values.mean() if not values.empty else np.nan,
                }
            )

    numeric_summary = pd.DataFrame(numeric_rows)
    numeric_path = REPORT_DIR / "numeric_features_by_error.csv"
    numeric_summary.to_csv(numeric_path, index=False)

    # 3. Find segments with weak late-order detection.
    # Include categorical input features and purchase month.
    categorical_features = [
        col
        for col in data.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns
        if col not in EXCLUDED_FEATURES
    ]

    segment_data = data.copy()

    if "order_purchase_timestamp" in segment_data.columns:
        segment_data["purchase_month"] = (
            pd.to_datetime(
                segment_data["order_purchase_timestamp"],
                errors="coerce",
            )
            .dt.to_period("M")
            .astype("string")
        )
        categorical_features.append("purchase_month")

    segment_rows = []

    for feature in categorical_features:
        # Avoid grouping by near-unique labels or identifiers.
        if segment_data[feature].nunique(dropna=False) > 50:
            continue

        grouped_values = segment_data[feature].astype("string").fillna("(Missing)")

        for value, group in segment_data.groupby(grouped_values, dropna=False):
            actual = group["is_late"]
            predicted = group["predicted_is_late"]

            tp = int(((actual == 1) & (predicted == 1)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())

            actual_late = int(actual.sum())
            alerts = int(predicted.sum())

            segment_rows.append(
                {
                    "feature": feature,
                    "segment": str(value),
                    "orders": len(group),
                    "actual_late_orders": actual_late,
                    "late_rate": actual_late / len(group),
                    "alerts": alerts,
                    "precision": tp / alerts if alerts else 0.0,
                    "recall": tp / actual_late if actual_late else np.nan,
                    "false_positives": fp,
                    "false_negatives": fn,
                }
            )

    segment_summary = pd.DataFrame(segment_rows)
    segment_path = REPORT_DIR / "error_analysis_segments.csv"
    segment_summary.to_csv(segment_path, index=False)

    # Show adequately sized segments with at least 10 late orders.
    if not segment_summary.empty:
        candidates = segment_summary[
            (segment_summary["orders"] >= 100)
            & (segment_summary["actual_late_orders"] >= 10)
        ].sort_values("recall", ascending=True)

        print("\nSEGMENTS WITH THE LOWEST LATE-ORDER RECALL")
        if candidates.empty:
            print("No segments met the minimum sample-size criteria.")
        else:
            print(
                candidates.head(15).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    print("\nSaved diagnostic reports:")
    print(outcome_path)
    print(numeric_path)
    print(segment_path)


if __name__ == "__main__":
    main()
