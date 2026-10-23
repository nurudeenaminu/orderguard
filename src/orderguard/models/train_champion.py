from pathlib import Path
from math import ceil

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "orderguard_champion.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "champion_validation_metrics.txt"
ALERT_REPORT_PATH = PROJECT_ROOT / "reports" / "champion_validation_alert_budgets.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "champion_validation_predictions.csv"

FEATURES = [
    "customer_state",
    "dominant_seller_state",
    "item_price_mean",
    "item_price_max",
    "seller_customer_same_state_fraction",
    "distinct_sellers",
    "purchase_month",
]

RECENCY_HALF_LIFE_DAYS = 90
ALERT_SHARES = [0.05, 0.10, 0.15, 0.20]


def is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def build_pipeline(X):
    categorical_features = [column for column in X.columns if is_categorical(X[column])]
    numeric_features = [
        column for column in X.columns if column not in categorical_features
    ]

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
        -age_days / RECENCY_HALF_LIFE_DAYS,
    )

    return weights / weights.mean()


def evaluate_alert_budgets(y_true, probabilities):
    """Evaluate ranking-based alert budgets without tuning a score cutoff."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    n = len(y_true)
    order = np.argsort(-probabilities, kind="stable")
    rows = []

    for share in ALERT_SHARES:
        alert_count = min(n, ceil(n * share))

        predictions = np.zeros(n, dtype=int)
        predictions[order[:alert_count]] = 1

        true_positives = int(((y_true == 1) & (predictions == 1)).sum())
        false_positives = int(((y_true == 0) & (predictions == 1)).sum())
        false_negatives = int(((y_true == 1) & (predictions == 0)).sum())

        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        rows.append(
            {
                "alert_budget_percent": share * 100,
                "orders_flagged": alert_count,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "precision_lift_over_prevalence": (
                    precision / y_true.mean() if y_true.mean() > 0 else np.nan
                ),
                "minimum_selected_score": (probabilities[order[alert_count - 1]]),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("Loading training and validation data...")

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

    for name, frame in [
        ("training", train),
        ("validation", validation),
    ]:
        missing = (set(FEATURES) | {"order_purchase_timestamp", "is_late"}) - set(
            frame.columns
        )

        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")

    X_train = train[FEATURES]
    y_train = train["is_late"].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation["is_late"].astype(int)

    weights = calculate_recency_weights(train["order_purchase_timestamp"])

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training late rate: {y_train.mean():.2%}")
    print(f"Validation late rate: {y_validation.mean():.2%}")
    print(f"Features: {len(FEATURES)}")
    print(f"Recency half-life: {RECENCY_HALF_LIFE_DAYS} days")

    model = build_pipeline(X_train)

    print("\nTraining champion logistic regression...")

    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=weights,
    )

    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X_validation)[:, positive_index]

    metrics = {
        "Model": "Logistic Regression with 90-day recency weighting",
        "Features": len(FEATURES),
        "Training orders": len(train),
        "Validation orders": len(validation),
        "Training late-order rate": y_train.mean(),
        "Validation late-order rate": y_validation.mean(),
        "Mean predicted probability": probabilities.mean(),
        "Average precision": average_precision_score(y_validation, probabilities),
        "Brier score": brier_score_loss(y_validation, probabilities),
        "Log loss": log_loss(
            y_validation,
            np.clip(probabilities, 1e-12, 1 - 1e-12),
            labels=[0, 1],
        ),
    }

    alert_budgets = evaluate_alert_budgets(
        y_validation,
        probabilities,
    )

    predictions = pd.DataFrame(
        {
            "order_id": validation["order_id"],
            "order_purchase_timestamp": validation["order_purchase_timestamp"],
            "actual_is_late": y_validation,
            "predicted_score": probabilities,
        }
    )

    predictions["risk_rank"] = (
        predictions["predicted_score"].rank(method="first", ascending=False).astype(int)
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the fitted preprocessing + estimator pipeline.
    joblib.dump(model, ARTIFACT_PATH)

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    alert_budgets.to_csv(ALERT_REPORT_PATH, index=False)

    report = "\n".join(
        (
            f"{key}: {value:.4f}"
            if isinstance(value, (float, np.floating))
            else f"{key}: {value}"
        )
        for key, value in metrics.items()
    )

    METRICS_PATH.write_text(report, encoding="utf-8")

    print("\nCHAMPION MODEL METRICS")
    print(report)

    print("\nVALIDATION ALERT-BUDGET COMPARISON")
    print(
        alert_budgets.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nModel saved to: {ARTIFACT_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Alert report saved to: {ALERT_REPORT_PATH}")
    print(f"Validation predictions saved to: {PREDICTIONS_PATH}")

    print(
        "\nImportant: validation has already been used in prior "
        "experiments. These results are development diagnostics, "
        "not an independent final estimate of future performance."
    )


if __name__ == "__main__":
    main()
