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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_set_comparison.csv"

THRESHOLD = 0.50

EXCLUDED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
]

TOP_EIGHT_FEATURES = [
    "customer_state",
    "dominant_seller_state",
    "item_price_mean",
    "item_price_max",
    "payment_value_sum",
    "seller_customer_same_state_fraction",
    "distinct_sellers",
    "purchase_month",
]

GEOGRAPHIC_FEATURES = [
    "customer_state",
    "dominant_seller_state",
    "seller_customer_same_state_fraction",
]


def is_categorical(series):
    """Identify columns requiring categorical preprocessing."""
    dtype = series.dtype

    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def build_pipeline(X):
    """Build identical preprocessing and classifier for each experiment."""
    categorical_features = [column for column in X.columns if is_categorical(X[column])]

    numeric_features = [
        column for column in X.columns if column not in categorical_features
    ]

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
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def best_validation_f1(y_true, probabilities):
    """Descriptive threshold search on validation data only."""
    thresholds = np.arange(0.01, 1.00, 0.01)

    scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(scores))

    return thresholds[best_index], scores[best_index]


def main():
    print("Loading training and validation data...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    y_train = train["is_late"].astype(int)
    y_validation = validation["is_late"].astype(int)

    X_train_full = train.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )
    X_validation_full = validation.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    if list(X_train_full.columns) != list(X_validation_full.columns):
        raise ValueError("Training and validation feature columns do not match.")

    missing_top_features = set(TOP_EIGHT_FEATURES) - set(X_train_full.columns)

    if missing_top_features:
        raise ValueError(f"Expected features not found: {sorted(missing_top_features)}")

    all_features = list(X_train_full.columns)

    feature_sets = {
        "A_full_baseline": all_features,
        "B_without_dominant_category": [
            feature for feature in all_features if feature != "dominant_category"
        ],
        "C_top_eight_features": TOP_EIGHT_FEATURES,
        "D_without_geography": [
            feature for feature in all_features if feature not in GEOGRAPHIC_FEATURES
        ],
    }

    results = []

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation late-order rate: {y_validation.mean():.2%}")

    for experiment_name, features in feature_sets.items():
        print(f"\nRunning {experiment_name} ({len(features)} features)...")

        X_train = X_train_full[features]
        X_validation = X_validation_full[features]

        model = build_pipeline(X_train)
        model.fit(X_train, y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_validation)[:, positive_index]

        # Compare at the same default threshold across experiments.
        predictions = (probabilities >= THRESHOLD).astype(int)

        ap = average_precision_score(y_validation, probabilities)
        precision = precision_score(y_validation, predictions, zero_division=0)
        recall = recall_score(y_validation, predictions, zero_division=0)
        f1 = f1_score(y_validation, predictions, zero_division=0)

        best_threshold, best_f1 = best_validation_f1(y_validation, probabilities)

        results.append(
            {
                "experiment": experiment_name,
                "feature_count": len(features),
                "average_precision": ap,
                "precision_at_0_50": precision,
                "recall_at_0_50": recall,
                "f1_at_0_50": f1,
                "best_validation_threshold": best_threshold,
                "best_validation_f1": best_f1,
            }
        )

    summary = pd.DataFrame(results).sort_values(
        "average_precision",
        ascending=False,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_PATH, index=False)

    print("\nFEATURE-SET COMPARISON")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nResults saved to:\n{REPORT_PATH}")
    print(
        "\nSelection note: average precision is the main ranking metric. "
        "The best validation F1 and threshold are descriptive only; "
        "they are not independent estimates of generalization."
    )


if __name__ == "__main__":
    main()
