from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "logistic_regression_baseline.joblib"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "validation_permutation_importance.csv"

EXCLUDED_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
]


def main():
    print("Loading logistic regression and validation data...")

    model = joblib.load(MODEL_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    y = validation["is_late"].astype(int)
    X = validation.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]

    baseline_ap = average_precision_score(y, probabilities)

    print(f"Validation average precision: {baseline_ap:.4f}")
    print(f"Features evaluated: {X.shape[1]}")
    print("\nCalculating permutation importance...")
    print("This may take a little while.")

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_ap_decrease": result.importances_mean,
            "std_ap_decrease": result.importances_std,
        }
    )

    importance = importance.sort_values(
        "mean_ap_decrease",
        ascending=False,
    ).reset_index(drop=True)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORT_PATH, index=False)

    print("\nTOP 15 FEATURES BY PERMUTATION IMPORTANCE")
    print(
        importance.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nFEATURES WITH NEGATIVE MEAN IMPORTANCE")
    negative = importance[importance["mean_ap_decrease"] < 0]

    if negative.empty:
        print("None.")
    else:
        print(
            negative.to_string(
                index=False,
                float_format=lambda value: f"{value:.5f}",
            )
        )

    print(f"\nFull results saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
