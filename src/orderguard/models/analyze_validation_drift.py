from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "logistic_regression_baseline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"

THRESHOLD = 0.62

EXCLUDED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
]


def monthly_late_rates(train, validation):
    """Compare monthly late-delivery rates across development splits."""
    frames = []

    for split_name, frame in [
        ("train", train),
        ("validation", validation),
    ]:
        temp = frame.copy()
        temp["purchase_month"] = (
            pd.to_datetime(temp["order_purchase_timestamp"], errors="coerce")
            .dt.to_period("M")
            .astype("string")
        )

        summary = (
            temp.groupby("purchase_month", dropna=False)
            .agg(
                orders=("is_late", "size"),
                late_orders=("is_late", "sum"),
            )
            .reset_index()
        )

        summary["split"] = split_name
        summary["late_rate"] = summary["late_orders"] / summary["orders"]
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def feature_drift(train_x, validation_x):
    """Compare feature distributions without using the test set."""
    rows = []

    for feature in train_x.columns:
        train_col = train_x[feature]
        val_col = validation_x[feature]

        row = {
            "feature": feature,
            "train_missing_rate": train_col.isna().mean(),
            "validation_missing_rate": val_col.isna().mean(),
            "train_unique_values": train_col.nunique(dropna=True),
            "validation_unique_values": val_col.nunique(dropna=True),
        }

        if is_numeric_dtype(train_col):
            train_values = train_col.dropna()
            val_values = val_col.dropna()

            train_mean = train_values.mean()
            train_std = train_values.std()

            row.update(
                {
                    "feature_type": "numeric",
                    "train_mean": train_mean,
                    "validation_mean": val_values.mean(),
                    "train_median": train_values.median(),
                    "validation_median": val_values.median(),
                    "mean_shift_in_train_std": (
                        (val_values.mean() - train_mean) / train_std
                        if pd.notna(train_std) and train_std > 0
                        else np.nan
                    ),
                }
            )

        else:
            train_values = train_col.astype("string").fillna("(Missing)")
            val_values = val_col.astype("string").fillna("(Missing)")

            train_freq = train_values.value_counts(normalize=True)
            val_freq = val_values.value_counts(normalize=True)

            categories = train_freq.index.union(val_freq.index)

            # Total variation distance: 0 means identical proportions;
            # 1 means completely non-overlapping distributions.
            total_variation = (
                train_freq.reindex(categories, fill_value=0)
                .sub(val_freq.reindex(categories, fill_value=0))
                .abs()
                .sum()
                / 2
            )

            known_categories = set(train_values.unique())
            unseen_rate = (~val_values.isin(known_categories)).mean()

            row.update(
                {
                    "feature_type": "categorical",
                    "train_mean": np.nan,
                    "validation_mean": np.nan,
                    "train_median": np.nan,
                    "validation_median": np.nan,
                    "mean_shift_in_train_std": np.nan,
                    "categorical_total_variation": total_variation,
                    "unseen_validation_category_rate": unseen_rate,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def monthly_validation_performance(validation, model):
    """Evaluate the locked threshold separately by validation month."""
    y = validation["is_late"].astype(int)

    X = validation.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    predictions = (probabilities >= THRESHOLD).astype(int)

    temp = validation[["order_purchase_timestamp", "is_late"]].copy()

    temp["purchase_month"] = (
        pd.to_datetime(temp["order_purchase_timestamp"], errors="coerce")
        .dt.to_period("M")
        .astype("string")
    )

    temp["actual"] = y.to_numpy()
    temp["predicted"] = predictions
    temp["probability"] = probabilities

    rows = []

    for month, group in temp.groupby("purchase_month", dropna=False):
        actual = group["actual"]
        predicted = group["predicted"]

        rows.append(
            {
                "purchase_month": month,
                "orders": len(group),
                "actual_late_orders": int(actual.sum()),
                "late_rate": actual.mean(),
                "alerts": int(predicted.sum()),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1": f1_score(actual, predicted, zero_division=0),
                "average_precision": (
                    average_precision_score(actual, group["probability"])
                    if actual.nunique() > 1
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("Loading training and validation data...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    train_x = train.drop(columns=EXCLUDED_COLUMNS, errors="ignore")
    validation_x = validation.drop(columns=EXCLUDED_COLUMNS, errors="ignore")

    if list(train_x.columns) != list(validation_x.columns):
        raise ValueError("Training and validation feature columns do not match.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check whether the outcome rate changes over time.
    monthly_rates = monthly_late_rates(train, validation)
    monthly_rates.to_csv(
        REPORT_DIR / "train_validation_monthly_late_rates.csv",
        index=False,
    )

    print("\nMONTHLY LATE-DELIVERY RATES")
    print(
        monthly_rates[
            ["split", "purchase_month", "orders", "late_orders", "late_rate"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # 2. Compare feature distributions.
    drift = feature_drift(train_x, validation_x)
    drift.to_csv(
        REPORT_DIR / "train_validation_feature_drift.csv",
        index=False,
    )

    print("\nLARGEST NUMERIC MEAN SHIFTS")
    numeric_drift = drift[drift["feature_type"] == "numeric"].copy()

    numeric_drift["absolute_shift"] = numeric_drift["mean_shift_in_train_std"].abs()

    print(
        numeric_drift.sort_values("absolute_shift", ascending=False)[
            ["feature", "train_mean", "validation_mean", "mean_shift_in_train_std"]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nLARGEST CATEGORICAL DISTRIBUTION SHIFTS")
    categorical_drift = drift[drift["feature_type"] == "categorical"].copy()

    print(
        categorical_drift.sort_values(
            "categorical_total_variation",
            ascending=False,
        )[["feature", "categorical_total_variation", "unseen_validation_category_rate"]]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # 3. See whether validation performance varies by month.
    model = joblib.load(MODEL_PATH)
    monthly_performance = monthly_validation_performance(validation, model)

    monthly_performance.to_csv(
        REPORT_DIR / "validation_monthly_model_performance.csv",
        index=False,
    )

    print("\nVALIDATION PERFORMANCE BY MONTH")
    print(
        monthly_performance.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved reports:")
    print(REPORT_DIR / "train_validation_monthly_late_rates.csv")
    print(REPORT_DIR / "train_validation_feature_drift.csv")
    print(REPORT_DIR / "validation_monthly_model_performance.csv")


if __name__ == "__main__":
    main()
