from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = REPORT_DIR / "temporal_calibration_comparison.csv"
BINS_PATH = REPORT_DIR / "temporal_calibration_bins.csv"

FEATURES = [
    "customer_state",
    "dominant_seller_state",
    "item_price_mean",
    "item_price_max",
    "payment_value_sum",
    "seller_customer_same_state_fraction",
    "distinct_sellers",
    "purchase_month",
]

N_BINS = 10


def is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def build_pipeline(X):
    categorical_features = [col for col in X.columns if is_categorical(X[col])]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    return Pipeline(
        [
            ("preprocessing", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def logit(probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-8,
        1 - 1e-8,
    )
    return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)


def expected_calibration_error(y_true, probabilities):
    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "probability": np.asarray(probabilities),
        }
    )

    frame["bin"] = pd.qcut(
        frame["probability"],
        q=N_BINS,
        labels=False,
        duplicates="drop",
    )

    bins = (
        frame.groupby("bin", observed=True)
        .agg(
            orders=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_late_rate=("actual", "mean"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
        )
        .reset_index()
    )

    weights = bins["orders"] / bins["orders"].sum()

    ece = (
        weights
        * (bins["mean_predicted_probability"] - bins["observed_late_rate"]).abs()
    ).sum()

    return float(ece), bins


def evaluate(y_true, probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-8,
        1 - 1e-8,
    )

    ece, bins = expected_calibration_error(y_true, probabilities)

    metrics = {
        "observed_late_rate": np.mean(y_true),
        "mean_predicted_probability": probabilities.mean(),
        "mean_probability_minus_observed_rate": (
            probabilities.mean() - np.mean(y_true)
        ),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "expected_calibration_error": ece,
    }

    return metrics, bins


def main():
    print("Loading development data...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    train["order_purchase_timestamp"] = pd.to_datetime(
        train["order_purchase_timestamp"],
        errors="raise",
    )
    validation["order_purchase_timestamp"] = pd.to_datetime(
        validation["order_purchase_timestamp"],
        errors="raise",
    )

    train = train.sort_values("order_purchase_timestamp").reset_index(drop=True)

    missing = set(FEATURES) - set(train.columns)
    if missing:
        raise ValueError(f"Missing training features: {sorted(missing)}")

    missing = set(FEATURES) - set(validation.columns)
    if missing:
        raise ValueError(f"Missing validation features: {sorted(missing)}")

    # Chronological split: early training block, then calibration block.
    split_index = int(len(train) * 0.80)

    base_train = train.iloc[:split_index].copy()
    calibration = train.iloc[split_index:].copy()

    X_base = base_train[FEATURES]
    y_base = base_train["is_late"].astype(int)

    X_cal = calibration[FEATURES]
    y_cal = calibration["is_late"].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation["is_late"].astype(int)

    print(f"Base-training rows: {len(base_train):,}")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Validation rows: {len(validation):,}")

    print(f"Base-training late rate: {y_base.mean():.2%}")
    print(f"Calibration late rate: {y_cal.mean():.2%}")
    print(f"Validation late rate: {y_validation.mean():.2%}")

    print("\nTraining unweighted logistic regression...")

    model = build_pipeline(X_base)
    model.fit(X_base, y_base)

    positive_index = list(model.classes_).index(1)

    calibration_probabilities = model.predict_proba(X_cal)[:, positive_index]

    validation_probabilities = model.predict_proba(X_validation)[:, positive_index]

    print("\nFitting sigmoid calibration on the later training block...")

    calibrator = LogisticRegression(
        C=1e6,
        max_iter=2000,
        random_state=42,
    )

    calibrator.fit(logit(calibration_probabilities), y_cal)

    calibrated_probabilities = calibrator.predict_proba(
        logit(validation_probabilities)
    )[:, list(calibrator.classes_).index(1)]

    calibration_slope = float(calibrator.coef_[0, 0])
    calibration_intercept = float(calibrator.intercept_[0])

    print(f"Calibration slope: {calibration_slope:.4f}")
    print(f"Calibration intercept: {calibration_intercept:.4f}")

    if calibration_slope <= 0:
        print(
            "WARNING: The calibration slope is non-positive. "
            "The score-to-risk relationship may not transfer reliably."
        )

    raw_metrics, raw_bins = evaluate(
        y_validation,
        validation_probabilities,
    )
    raw_metrics["strategy"] = "Uncalibrated"
    raw_bins["strategy"] = "Uncalibrated"

    calibrated_metrics, calibrated_bins = evaluate(
        y_validation,
        calibrated_probabilities,
    )
    calibrated_metrics["strategy"] = "Temporal sigmoid calibration"
    calibrated_bins["strategy"] = "Temporal sigmoid calibration"

    summary = pd.DataFrame(
        [
            raw_metrics,
            calibrated_metrics,
        ]
    )

    summary = summary[
        ["strategy"] + [col for col in summary.columns if col != "strategy"]
    ]

    bins = pd.concat(
        [raw_bins, calibrated_bins],
        ignore_index=True,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    bins.to_csv(BINS_PATH, index=False)

    print("\nVALIDATION CALIBRATION COMPARISON")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nVALIDATION CALIBRATION BINS")
    print(
        bins[
            [
                "strategy",
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
