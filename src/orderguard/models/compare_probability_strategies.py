from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = REPORT_DIR / "probability_strategy_comparison.csv"
BINS_PATH = REPORT_DIR / "probability_strategy_calibration_bins.csv"

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

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def build_pipeline(X, class_weight):
    categorical_features = [col for col in X.columns if is_categorical(X[col])]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessing = ColumnTransformer(transformers=transformers)

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def calibration_bins(y_true, probabilities, strategy):
    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true),
            "probability": np.asarray(probabilities),
        }
    )

    frame["bin"] = pd.qcut(
        frame["probability"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    result = (
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

    result.insert(0, "strategy", strategy)

    weights = result["orders"] / result["orders"].sum()
    result["weighted_absolute_gap"] = (
        weights
        * (result["mean_predicted_probability"] - result["observed_late_rate"]).abs()
    )

    return result


def evaluate_probabilities(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-12,
        1 - 1e-12,
    )

    bins = calibration_bins(
        y_true,
        probabilities,
        strategy="temporary",
    )

    ece = bins["weighted_absolute_gap"].sum()

    threshold_rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )

    threshold_table = pd.DataFrame(threshold_rows)
    best = threshold_table.loc[threshold_table["f1"].idxmax()]

    predictions_at_half = (probabilities >= 0.50).astype(int)

    return {
        "observed_late_rate": y_true.mean(),
        "mean_predicted_probability": probabilities.mean(),
        "mean_probability_minus_observed_rate": (probabilities.mean() - y_true.mean()),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "expected_calibration_error": ece,
        "precision_at_0_50": precision_score(
            y_true, predictions_at_half, zero_division=0
        ),
        "recall_at_0_50": recall_score(y_true, predictions_at_half, zero_division=0),
        "f1_at_0_50": f1_score(y_true, predictions_at_half, zero_division=0),
        "best_validation_f1_threshold": best["threshold"],
        "best_validation_f1": best["f1"],
        "precision_at_best_f1": best["precision"],
        "recall_at_best_f1": best["recall"],
    }


def correct_balanced_probabilities(
    probabilities,
    training_late_rate,
):
    """
    Approximate prior correction for balanced class weights.

    Assumes weighting mainly shifts the class odds.
    It is not guaranteed to produce calibrated probabilities.
    """
    p = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-12,
        1 - 1e-12,
    )
    prevalence = training_late_rate

    corrected = p * prevalence / (p * prevalence + (1 - p) * (1 - prevalence))

    return corrected


def main():
    print("Loading training and validation data...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    missing = set(FEATURES) - set(train.columns)
    if missing:
        raise ValueError(f"Training data is missing features: {sorted(missing)}")

    missing_validation = set(FEATURES) - set(validation.columns)
    if missing_validation:
        raise ValueError(
            f"Validation data is missing features: " f"{sorted(missing_validation)}"
        )

    X_train = train[FEATURES]
    y_train = train["is_late"].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation["is_late"].astype(int)

    training_late_rate = y_train.mean()

    print(f"Training late-order rate: {training_late_rate:.2%}")
    print(f"Validation late-order rate: {y_validation.mean():.2%}")

    strategies = [
        ("A_unweighted", None),
        ("B_balanced", "balanced"),
    ]

    all_results = []
    all_bins = []

    for strategy, class_weight in strategies:
        print(f"\nTraining {strategy}...")

        model = build_pipeline(X_train, class_weight)
        model.fit(X_train, y_train)

        positive_index = list(model.classes_).index(1)

        raw_probabilities = model.predict_proba(X_validation)[:, positive_index]

        probability_variants = {
            strategy + "_raw": raw_probabilities,
        }

        if strategy == "B_balanced":
            probability_variants["C_balanced_prior_corrected"] = (
                correct_balanced_probabilities(
                    raw_probabilities,
                    training_late_rate,
                )
            )

        for name, probabilities in probability_variants.items():
            print(f"Evaluating {name}...")

            metrics = evaluate_probabilities(
                y_validation,
                probabilities,
            )
            metrics["strategy"] = name
            all_results.append(metrics)

            bins = calibration_bins(
                y_validation,
                probabilities,
                strategy=name,
            )
            all_bins.append(bins)

    summary = pd.DataFrame(all_results)

    summary = summary[
        ["strategy"] + [col for col in summary.columns if col != "strategy"]
    ]

    bins_summary = pd.concat(all_bins, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    bins_summary.to_csv(BINS_PATH, index=False)

    print("\nPROBABILITY STRATEGY COMPARISON")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nCALIBRATION BINS")
    print(
        bins_summary[
            [
                "strategy",
                "bin",
                "orders",
                "mean_predicted_probability",
                "observed_late_rate",
                "weighted_absolute_gap",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Calibration bins saved to: {BINS_PATH}")

    print(
        "\nImportant: the best F1 thresholds are selected on the "
        "same validation set and are descriptive, not unbiased "
        "estimates of future performance."
    )


if __name__ == "__main__":
    main()
