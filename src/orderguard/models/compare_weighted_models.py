from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

FOLD_REPORT = REPORT_DIR / "weighted_model_comparison_folds.csv"
SUMMARY_REPORT = REPORT_DIR / "weighted_model_comparison_summary.csv"

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


def build_pipeline(X, algorithm):
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

    if algorithm == "Logistic Regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=42,
        )
    elif algorithm == "Gradient Boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=100,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return Pipeline(
        [
            ("preprocessing", ColumnTransformer(transformers=transformers)),
            ("classifier", classifier),
        ]
    )


def calculate_recency_weights(timestamps):
    latest = timestamps.max()

    age_days = (latest - timestamps).dt.total_seconds() / 86400

    weights = np.power(2.0, -age_days / HALF_LIFE_DAYS)

    return weights / weights.mean()


def prepare_features(frame, categorical_month):
    X = frame[FEATURES].copy()

    if categorical_month:
        X["purchase_month"] = X["purchase_month"].astype(str)

    return X


def main():
    print("Loading and chronologically sorting training data...")

    data = pd.read_csv(TRAIN_PATH)
    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="raise",
    )
    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)

    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Required features missing: {sorted(missing)}")

    y = data["is_late"].astype(int)
    dates = data["order_purchase_timestamp"]

    experiments = [
        ("LR_numeric_unweighted", "Logistic Regression", False, False),
        ("LR_numeric_recency90", "Logistic Regression", False, True),
        ("LR_categorical_recency90", "Logistic Regression", True, True),
        ("GB_numeric_recency90", "Gradient Boosting", False, True),
        ("GB_categorical_recency90", "Gradient Boosting", True, True),
    ]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data), start=1):
        train = data.iloc[train_idx]
        validation = data.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[val_idx]
        train_dates = dates.iloc[train_idx]

        print(f"\nFold {fold}")
        print(f"Training ends: {train_dates.max()}")
        print(
            f"Validation period: {dates.iloc[val_idx].min()} "
            f"to {dates.iloc[val_idx].max()}"
        )
        print(f"Validation late rate: {y_validation.mean():.2%}")

        for name, algorithm, categorical_month, use_recency in experiments:
            print(f"  Running {name}...")

            X_train = prepare_features(train, categorical_month)
            X_validation = prepare_features(validation, categorical_month)

            model = build_pipeline(X_train, algorithm)

            if use_recency:
                sample_weights = calculate_recency_weights(train_dates)
                model.fit(
                    X_train,
                    y_train,
                    classifier__sample_weight=sample_weights,
                )
            else:
                model.fit(X_train, y_train)

            positive_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(X_validation)[:, positive_index]

            ap = average_precision_score(y_validation, probabilities)

            rows.append(
                {
                    "fold": fold,
                    "experiment": name,
                    "algorithm": algorithm,
                    "month_encoding": (
                        "categorical" if categorical_month else "numeric"
                    ),
                    "recency_half_life_days": (HALF_LIFE_DAYS if use_recency else 0),
                    "validation_rows": len(validation),
                    "late_rate": y_validation.mean(),
                    "average_precision": ap,
                    "ap_over_prevalence": ap / y_validation.mean(),
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
        .sort_values("mean_average_precision", ascending=False)
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

    print("\nMODEL COMPARISON SUMMARY")
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
