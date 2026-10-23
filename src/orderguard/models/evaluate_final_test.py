from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "logistic_regression_baseline.joblib"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "final_test_metrics.txt"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"

# Locked using validation results, not test results.
THRESHOLD = 0.62

EXCLUDED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
]


def main():
    print("Loading selected model and test data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(TEST_PATH)

    y_true = test["is_late"].astype(int)

    X_test = test.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    positive_class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X_test)[:, positive_class_index]

    y_pred = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "Model": "Logistic Regression",
        "Threshold (selected on validation)": THRESHOLD,
        "Test orders": len(test),
        "Actual late orders": int(y_true.sum()),
        "Actual late-order rate": y_true.mean(),
        "Orders flagged": int(y_pred.sum()),
        "Fraction of orders flagged": y_pred.mean(),
        "True positives": int(tp),
        "False positives": int(fp),
        "True negatives": int(tn),
        "False negatives": int(fn),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Average precision (PR-AUC)": average_precision_score(y_true, probabilities),
    }

    # Create an auditable record of each test prediction.
    predictions = pd.DataFrame(
        {
            "order_id": test["order_id"],
            "order_purchase_timestamp": test["order_purchase_timestamp"],
            "actual_is_late": y_true,
            "predicted_late_probability": probabilities,
            "predicted_is_late": y_pred,
        }
    )

    predictions["prediction_outcome"] = np.select(
        [
            (y_true.to_numpy() == 1) & (y_pred == 1),
            (y_true.to_numpy() == 0) & (y_pred == 1),
            (y_true.to_numpy() == 1) & (y_pred == 0),
        ],
        [
            "true_positive",
            "false_positive",
            "false_negative",
        ],
        default="true_negative",
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    lines = [
        "OrderGuard — Final Test Evaluation",
        "=" * 40,
        "",
        "Threshold selected on validation data: 0.62",
        "Confusion matrix: rows = actual [on-time, late];",
        "columns = predicted [on-time, late]",
        str(np.array([[tn, fp], [fn, tp]])),
        "",
    ]

    for name, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            lines.append(f"{name}: {value:.4f}")
        else:
            lines.append(f"{name}: {value}")

    report_text = "\n".join(lines)

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nMetrics saved to: {REPORT_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
