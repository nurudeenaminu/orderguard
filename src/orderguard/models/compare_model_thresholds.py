from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

REPORT_PATH = PROJECT_ROOT / "reports" / "model_threshold_comparison.csv"

MODEL_PATHS = {
    "Logistic Regression": (
        PROJECT_ROOT / "artifacts" / "logistic_regression_baseline.joblib"
    ),
    "Gradient Boosting": (
        PROJECT_ROOT / "artifacts" / "gradient_boosting_baseline.joblib"
    ),
}

# Metadata, target, and the feature that has not yet passed
# our availability-at-prediction-time audit.
EXCLUDED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
]

# Explore thresholds from 0.01 through 0.99.
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def evaluate_model(name, model, X, y):
    """Evaluate a fitted model across classification thresholds."""

    print(f"\nEvaluating {name}...")

    # Get the predicted probability for the positive class (late = 1).
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_class_index]

    average_precision = average_precision_score(y, probabilities)

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        # Explicit label order: on-time (0), late (1).
        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

        rows.append(
            {
                "model": name,
                "threshold": threshold,
                "average_precision": average_precision,
                "alerts": int(tp + fp),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows), probabilities


def main():
    print("Loading validation data...")

    validation = pd.read_csv(VALIDATION_PATH)

    y = validation["is_late"].astype(int)

    X = validation.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    print(f"Validation orders: {len(validation):,}")
    print(f"Late orders: {int(y.sum()):,}")
    print(f"Actual late-order rate: {y.mean():.2%}")

    all_results = []
    model_summaries = []

    for name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        model = joblib.load(model_path)

        results, probabilities = evaluate_model(
            name=name,
            model=model,
            X=X,
            y=y,
        )

        all_results.append(results)

        # Best F1 is a comparison point, not an automatic
        # deployment decision.
        best_f1_row = results.loc[results["f1"].idxmax()]

        # Among thresholds that catch at least half of late orders,
        # identify the one with the highest precision.
        recall_candidates = results[results["recall"] >= 0.50]

        if not recall_candidates.empty:
            best_recall_row = recall_candidates.loc[
                recall_candidates["precision"].idxmax()
            ]
            recall_threshold = best_recall_row["threshold"]
            recall_precision = best_recall_row["precision"]
            recall_value = best_recall_row["recall"]
            recall_alerts = best_recall_row["alerts"]
        else:
            recall_threshold = np.nan
            recall_precision = np.nan
            recall_value = np.nan
            recall_alerts = np.nan

        model_summaries.append(
            {
                "model": name,
                "average_precision": results["average_precision"].iloc[0],
                "min_probability": probabilities.min(),
                "max_probability": probabilities.max(),
                "mean_probability": probabilities.mean(),
                "best_f1_threshold": best_f1_row["threshold"],
                "best_f1": best_f1_row["f1"],
                "precision_at_best_f1": best_f1_row["precision"],
                "recall_at_best_f1": best_f1_row["recall"],
                "alerts_at_best_f1": best_f1_row["alerts"],
                "threshold_at_recall_50": recall_threshold,
                "precision_at_recall_50": recall_precision,
                "recall_at_recall_50": recall_value,
                "alerts_at_recall_50": recall_alerts,
            }
        )

    comparison = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(model_summaries)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORT_PATH, index=False)

    print("\nMODEL SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTOP 5 THRESHOLDS BY F1 FOR EACH MODEL")

    for name in MODEL_PATHS:
        model_results = comparison[comparison["model"] == name]

        top_f1 = model_results.nlargest(5, "f1")[
            [
                "threshold",
                "alerts",
                "precision",
                "recall",
                "f1",
                "average_precision",
            ]
        ]

        print(f"\n{name}")
        print(
            top_f1.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print(f"\nFull threshold results saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
