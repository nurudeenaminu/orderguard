from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

FOLD_REPORT_PATH = REPORT_DIR / "rolling_threshold_policy.csv"
PREDICTIONS_PATH = REPORT_DIR / "rolling_threshold_oof_predictions.csv"

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

N_SPLITS = 3
MINIMUM_RECALL = 0.50
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


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

    return Pipeline(
        [
            (
                "preprocessing",
                ColumnTransformer(transformers=transformers),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def select_threshold(y_true, probabilities):
    """Maximize precision, subject to recall being at least 50%."""
    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )

    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates["recall"] >= MINIMUM_RECALL]

    if eligible.empty:
        raise ValueError("No tested threshold achieved the minimum recall.")

    highest_precision = eligible["precision"].max()

    # If precision ties, prefer the higher threshold.
    best = (
        eligible[eligible["precision"] >= highest_precision - 1e-12]
        .sort_values("threshold", ascending=False)
        .iloc[0]
    )

    return float(best["threshold"])


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "alerts": int(predictions.sum()),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def main():
    print("Loading and sorting training data...")

    data = pd.read_csv(TRAIN_PATH)
    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="raise",
    )
    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)

    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Required features missing: {sorted(missing)}")

    X = data[FEATURES]
    y = data["is_late"].astype(int)
    timestamps = data["order_purchase_timestamp"]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    previous_predictions = []
    fold_results = []
    prediction_records = []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X), start=1):
        print(f"\nFold {fold}")

        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]
        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        print(
            f"Training period: {timestamps.iloc[train_idx].min()} "
            f"to {timestamps.iloc[train_idx].max()}"
        )
        print(
            f"Evaluation period: {timestamps.iloc[validation_idx].min()} "
            f"to {timestamps.iloc[validation_idx].max()}"
        )
        print(f"Evaluation late rate: {y_validation.mean():.2%}")

        model = build_pipeline(X_train)
        model.fit(X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_validation)[:, positive_index]

        # The first fold has no previous out-of-fold predictions
        # from which to learn a threshold.
        if not previous_predictions:
            selected_threshold = 0.50
            threshold_source = "default_0.50_no_prior_fold"
        else:
            past = pd.concat(
                previous_predictions,
                ignore_index=True,
            )

            selected_threshold = select_threshold(
                past["actual_is_late"].to_numpy(),
                past["predicted_probability"].to_numpy(),
            )
            threshold_source = "previous_folds"

        rolling_metrics = evaluate(
            y_validation.to_numpy(),
            probabilities,
            selected_threshold,
        )

        fixed_metrics = evaluate(
            y_validation.to_numpy(),
            probabilities,
            0.50,
        )

        ap = average_precision_score(y_validation, probabilities)

        result = {
            "fold": fold,
            "train_rows": len(train_idx),
            "evaluation_rows": len(validation_idx),
            "evaluation_start": timestamps.iloc[validation_idx].min(),
            "evaluation_end": timestamps.iloc[validation_idx].max(),
            "late_rate": y_validation.mean(),
            "average_precision": ap,
            "threshold_source": threshold_source,
            "selected_threshold": selected_threshold,
            "rolling_alerts": rolling_metrics["alerts"],
            "rolling_precision": rolling_metrics["precision"],
            "rolling_recall": rolling_metrics["recall"],
            "rolling_f1": rolling_metrics["f1"],
            "fixed_0_50_alerts": fixed_metrics["alerts"],
            "fixed_0_50_precision": fixed_metrics["precision"],
            "fixed_0_50_recall": fixed_metrics["recall"],
            "fixed_0_50_f1": fixed_metrics["f1"],
        }

        fold_results.append(result)

        print(f"Selected threshold: {selected_threshold:.2f}")
        print(
            f"Rolling policy — precision: "
            f"{rolling_metrics['precision']:.4f}, "
            f"recall: {rolling_metrics['recall']:.4f}, "
            f"F1: {rolling_metrics['f1']:.4f}, "
            f"alerts: {rolling_metrics['alerts']}"
        )
        print(
            f"Fixed 0.50 — precision: "
            f"{fixed_metrics['precision']:.4f}, "
            f"recall: {fixed_metrics['recall']:.4f}, "
            f"F1: {fixed_metrics['f1']:.4f}, "
            f"alerts: {fixed_metrics['alerts']}"
        )

        current_predictions = pd.DataFrame(
            {
                "fold": fold,
                "order_purchase_timestamp": timestamps.iloc[validation_idx].to_numpy(),
                "actual_is_late": y_validation.to_numpy(),
                "predicted_probability": probabilities,
            }
        )

        prediction_records.append(current_predictions)
        previous_predictions.append(current_predictions)

    results = pd.DataFrame(fold_results)
    predictions = pd.concat(prediction_records, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(FOLD_REPORT_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print("\nROLLING THRESHOLD RESULTS")
    print(
        results[
            [
                "fold",
                "late_rate",
                "average_precision",
                "selected_threshold",
                "rolling_alerts",
                "rolling_precision",
                "rolling_recall",
                "rolling_f1",
                "fixed_0_50_f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nFold report: {FOLD_REPORT_PATH}")
    print(f"OOF predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
