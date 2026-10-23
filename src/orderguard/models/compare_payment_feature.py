from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

FOLD_REPORT = REPORT_DIR / "payment_feature_ablation_folds.csv"
SUMMARY_REPORT = REPORT_DIR / "payment_feature_ablation_summary.csv"

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

HALF_LIFE_DAYS = 90
N_SPLITS = 3


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


def calculate_recency_weights(timestamps):
    latest_date = timestamps.max()

    age_days = (latest_date - timestamps).dt.total_seconds() / 86400

    weights = np.power(
        2.0,
        -age_days / HALF_LIFE_DAYS,
    )

    return weights / weights.mean()


def main():
    print("Loading and chronologically sorting training data...")

    data = pd.read_csv(TRAIN_PATH)

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="raise",
    )

    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)

    missing = set(
        FEATURES
        + [
            "order_purchase_timestamp",
            "is_late",
        ]
    ) - set(data.columns)

    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    y = data["is_late"].astype(int)
    timestamps = data["order_purchase_timestamp"]

    feature_sets = {
        "A_with_payment_value": FEATURES,
        "B_without_payment_value": [
            feature for feature in FEATURES if feature != "payment_value_sum"
        ],
    }

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(data),
        start=1,
    ):
        train = data.iloc[train_idx]
        validation = data.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[val_idx]

        train_dates = timestamps.iloc[train_idx]

        weights = calculate_recency_weights(train_dates)

        print(f"\nFold {fold}")
        print(f"Training rows: {len(train):,}")
        print(f"Validation rows: {len(validation):,}")
        print(f"Training ends: {train_dates.max()}")
        print(
            "Validation period: "
            f"{timestamps.iloc[val_idx].min()} to "
            f"{timestamps.iloc[val_idx].max()}"
        )
        print(f"Validation late rate: {y_validation.mean():.2%}")

        for experiment, selected_features in feature_sets.items():
            print(f"  Running {experiment}...")

            X_train = train[selected_features]
            X_validation = validation[selected_features]

            model = build_pipeline(X_train)

            model.fit(
                X_train,
                y_train,
                classifier__sample_weight=weights,
            )

            positive_index = list(model.classes_).index(1)

            probabilities = model.predict_proba(X_validation)[:, positive_index]

            ap = average_precision_score(
                y_validation,
                probabilities,
            )

            prevalence = y_validation.mean()

            rows.append(
                {
                    "fold": fold,
                    "experiment": experiment,
                    "feature_count": len(selected_features),
                    "validation_orders": len(validation),
                    "late_rate": prevalence,
                    "average_precision": ap,
                    "ap_over_prevalence": (
                        ap / prevalence if prevalence > 0 else np.nan
                    ),
                }
            )

    results = pd.DataFrame(rows)

    summary = (
        results.groupby("experiment")
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_ap_over_prevalence=("ap_over_prevalence", "mean"),
        )
        .reset_index()
        .sort_values(
            "mean_average_precision",
            ascending=False,
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(FOLD_REPORT, index=False)
    summary.to_csv(SUMMARY_REPORT, index=False)

    print("\nRESULTS BY FOLD")
    print(
        results[
            [
                "fold",
                "experiment",
                "late_rate",
                "average_precision",
                "ap_over_prevalence",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nPAYMENT FEATURE ABLATION SUMMARY")
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
