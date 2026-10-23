from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

EXCLUDED_COLUMNS = {
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
}


def main():
    print("Loading saved gradient boosting model and validation data...")

    model = joblib.load(ARTIFACT_DIR / "gradient_boosting_baseline.joblib")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_val = validation.drop(columns=list(EXCLUDED_COLUMNS & set(validation.columns)))
    y_val = validation["is_late"].astype(int)

    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"\nActual late-order rate: {y_val.mean():.2%}")
    print(f"Minimum predicted probability: {probabilities.min():.6f}")
    print(f"Maximum predicted probability: {probabilities.max():.6f}")
    print("Mean predicted probability: " f"{probabilities.mean():.6f}")
    print(
        "PR-AUC (average precision): "
        f"{average_precision_score(y_val, probabilities):.4f}"
    )

    rows = []

    for threshold in [
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        0.50,
    ]:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "alerts": int(predictions.sum()),
                "precision": precision_score(y_val, predictions, zero_division=0),
                "recall": recall_score(y_val, predictions, zero_division=0),
                "f1": f1_score(y_val, predictions, zero_division=0),
            }
        )

    results = pd.DataFrame(rows)

    print("\nThreshold comparison:")
    print(results.round(4).to_string(index=False))

    # Save results so we can compare thresholds later.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "gradient_boosting_thresholds.csv"
    results.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    # Show the confusion matrix at a lower threshold for inspection.
    threshold = 0.05
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_val, predictions, labels=[0, 1])

    print(f"\nConfusion matrix at threshold {threshold:.2f}:")
    print("Rows = actual [on-time, late]; columns = predicted [on-time, late]")
    print(matrix)


if __name__ == "__main__":
    main()
