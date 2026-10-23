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

FOLD_REPORT = REPORT_DIR / "month_encoding_folds.csv"
SUMMARY_REPORT = REPORT_DIR / "month_encoding_summary.csv"

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
HALF_LIFE_DAYS = 90


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


def recency_weights(timestamps):
    latest = timestamps.max()

    age_days = (latest - timestamps).dt.total_seconds() / 86400

    weights = np.power(2.0, -age_days / HALF_LIFE_DAYS)

    return weights / weights.mean()


def prepare_features(frame, categorical_month):
    X = frame[FEATURES].copy()

    if categorical_month:
        # Treat month numbers as categories, not ordered quantities.
        X["purchase_month"] = X["purchase_month"].astype(str)

    return X


def main():
    print("Loading training split...")

    data = pd.read_csv(TRAIN_PATH)
    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="raise",
    )
    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)

    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    y = data["is_late"].astype(int)
    timestamps = data["order_purchase_timestamp"]

    experiments = [
        ("A_numeric_month_unweighted", False, False),
        ("B_categorical_month_unweighted", True, False),
        ("C_numeric_month_recency90", False, True),
        ("D_categorical_month_recency90", True, True),
    ]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data), start=1):
        train = data.iloc[train_idx]
        validation = data.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[val_idx]

        train_dates = timestamps.iloc[train_idx]

        print(f"\nFold {fold}")
        print(f"Validation late rate: {y_validation.mean():.2%}")

        for name, categorical_month, use_recency in experiments:
            print(f"  Running {name}...")

            X_train = prepare_features(train, categorical_month)
            X_validation = prepare_features(validation, categorical_month)

            model = build_pipeline(X_train)

            if use_recency:
                weights = recency_weights(train_dates)
                model.fit(
                    X_train,
                    y_train,
                    classifier__sample_weight=weights,
                )
            else:
                model.fit(X_train, y_train)

            positive_index = list(model.classes_).index(1)

            probabilities = model.predict_proba(X_validation)[:, positive_index]

            ap = average_precision_score(y_validation, probabilities)

            results.append(
                {
                    "fold": fold,
                    "experiment": name,
                    "categorical_month": categorical_month,
                    "recency_half_life_days": (HALF_LIFE_DAYS if use_recency else 0),
                    "validation_orders": len(validation),
                    "late_rate": y_validation.mean(),
                    "average_precision": ap,
                    "ap_over_prevalence": ap / y_validation.mean(),
                }
            )

    results_df = pd.DataFrame(results)

    summary = (
        results_df.groupby("experiment")
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_ap_over_prevalence=("ap_over_prevalence", "mean"),
        )
        .reset_index()
        .sort_values("mean_ap_over_prevalence", ascending=False)
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(FOLD_REPORT, index=False)
    summary.to_csv(SUMMARY_REPORT, index=False)

    print("\nRESULTS BY FOLD")
    print(
        results_df[
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

    print("\nMONTH-ENCODING SUMMARY")
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
