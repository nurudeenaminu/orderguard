from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

FOLD_REPORT = REPORT_DIR / "recency_weighting_folds.csv"
SUMMARY_REPORT = REPORT_DIR / "recency_weighting_summary.csv"

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

HALF_LIVES_DAYS = [90, 180, 365]
N_SPLITS = 3
THRESHOLD = 0.50


def is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def build_pipeline(X):
    categorical = [col for col in X.columns if is_categorical(X[col])]
    numeric = [col for col in X.columns if col not in categorical]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
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


def calculate_recency_weights(timestamps, half_life_days):
    """Assign exponentially decaying weights to older training examples."""
    latest_date = timestamps.max()

    age_days = (latest_date - timestamps).dt.total_seconds() / (24 * 60 * 60)

    weights = np.power(2.0, -age_days / half_life_days)

    # Keep the average sample weight at 1 for numerical consistency.
    return weights / weights.mean()


def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    train_dates,
    X_validation,
    y_validation,
):
    if model_name == "Unweighted":
        model.fit(X_train, y_train)
    else:
        half_life_days = int(model_name.replace("Half-life ", "").replace(" days", ""))

        weights = calculate_recency_weights(
            train_dates,
            half_life_days,
        )

        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=weights,
        )

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_validation)[:, positive_index]

    predictions = (probabilities >= THRESHOLD).astype(int)

    ap = average_precision_score(y_validation, probabilities)
    f1 = f1_score(y_validation, predictions, zero_division=0)

    return {
        "model": model_name,
        "validation_orders": len(y_validation),
        "late_rate": y_validation.mean(),
        "average_precision": ap,
        "ap_over_prevalence": ap / y_validation.mean(),
        "f1_at_0_50": f1,
        "alerts_at_0_50": int(predictions.sum()),
    }


def main():
    print("Loading training data...")

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
    dates = data["order_purchase_timestamp"]

    experiments = ["Unweighted"] + [
        f"Half-life {days} days" for days in HALF_LIVES_DAYS
    ]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[val_idx]

        train_dates = dates.iloc[train_idx]

        print(f"\nFold {fold}")
        print(f"Training end: {train_dates.max()}")
        print(
            f"Validation period: {dates.iloc[val_idx].min()} "
            f"to {dates.iloc[val_idx].max()}"
        )
        print(f"Validation late rate: {y_validation.mean():.2%}")

        for experiment in experiments:
            print(f"  Evaluating {experiment}...")

            model = build_pipeline(X_train)

            metrics = evaluate_model(
                model_name=experiment,
                model=model,
                X_train=X_train,
                y_train=y_train,
                train_dates=train_dates,
                X_validation=X_validation,
                y_validation=y_validation,
            )

            metrics["fold"] = fold
            results.append(metrics)

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby("model")
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_ap_over_prevalence=("ap_over_prevalence", "mean"),
            mean_f1_at_0_50=("f1_at_0_50", "mean"),
        )
        .reset_index()
        .sort_values("mean_average_precision", ascending=False)
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(FOLD_REPORT, index=False)
    summary.to_csv(SUMMARY_REPORT, index=False)

    print("\nRESULTS BY FOLD")
    print(
        results_df[
            [
                "fold",
                "model",
                "late_rate",
                "average_precision",
                "ap_over_prevalence",
                "f1_at_0_50",
                "alerts_at_0_50",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nRECENCY-WEIGHTING SUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nDetailed results: {FOLD_REPORT}")
    print(f"Summary: {SUMMARY_REPORT}")


if __name__ == "__main__":
    main()
