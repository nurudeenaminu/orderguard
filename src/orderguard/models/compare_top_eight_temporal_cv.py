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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
FOLD_REPORT_PATH = PROJECT_ROOT / "reports" / "top_eight_temporal_cv_folds.csv"
SUMMARY_REPORT_PATH = PROJECT_ROOT / "reports" / "top_eight_temporal_cv_summary.csv"

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
THRESHOLD = 0.50


def is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def build_pipeline(X, classifier):
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
            ("classifier", classifier),
        ]
    )


def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=100,
            random_state=42,
        ),
    }


def main():
    print("Loading and chronologically sorting training data...")

    data = pd.read_csv(TRAIN_PATH)
    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="raise",
    )
    data = data.sort_values("order_purchase_timestamp").reset_index(drop=True)

    missing_features = set(FEATURES) - set(data.columns)
    if missing_features:
        raise ValueError(f"Required features are missing: {sorted(missing_features)}")

    X = data[FEATURES]
    y = data["is_late"].astype(int)
    timestamps = data["order_purchase_timestamp"]

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(f"\nFold {fold}")
        print(
            f"  Training: {timestamps.iloc[train_idx].min()} "
            f"to {timestamps.iloc[train_idx].max()}"
        )
        print(
            f"  Validation: {timestamps.iloc[val_idx].min()} "
            f"to {timestamps.iloc[val_idx].max()}"
        )
        print(f"  Validation late rate: {y_val.mean():.2%}")

        for name, classifier in make_models().items():
            print(f"  Training {name}...")

            model = build_pipeline(X_train, classifier)

            if name == "Gradient Boosting":
                sample_weights = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
                model.fit(
                    X_train,
                    y_train,
                    classifier__sample_weight=sample_weights,
                )
            else:
                model.fit(X_train, y_train)

            positive_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(X_val)[:, positive_index]

            predictions = (probabilities >= THRESHOLD).astype(int)

            rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_rows": len(train_idx),
                    "validation_rows": len(val_idx),
                    "validation_start": timestamps.iloc[val_idx].min(),
                    "validation_end": timestamps.iloc[val_idx].max(),
                    "validation_late_rate": y_val.mean(),
                    "average_precision": average_precision_score(y_val, probabilities),
                    "f1_at_0_50": f1_score(y_val, predictions, zero_division=0),
                }
            )

    results = pd.DataFrame(rows)

    summary = (
        results.groupby("model")
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            min_average_precision=("average_precision", "min"),
            max_average_precision=("average_precision", "max"),
            mean_f1_at_0_50=("f1_at_0_50", "mean"),
        )
        .reset_index()
        .sort_values("mean_average_precision", ascending=False)
    )

    FOLD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(FOLD_REPORT_PATH, index=False)
    summary.to_csv(SUMMARY_REPORT_PATH, index=False)

    print("\nRESULTS BY FOLD")
    print(
        results[
            ["fold", "model", "validation_late_rate", "average_precision", "f1_at_0_50"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nTEMPORAL CROSS-VALIDATION SUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nDetailed results saved to: {FOLD_REPORT_PATH}")
    print(f"Summary saved to: {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
