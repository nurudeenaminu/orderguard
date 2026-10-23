from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_late"

# These columns are metadata, the target, or excluded pending a
# feature-availability audit.
EXCLUDED_COLUMNS = {
    "order_id",
    "order_purchase_timestamp",
    "is_late",
    "promised_delivery_days",
}


def main():
    print("1/5 Loading training and validation data...")

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train.drop(columns=list(EXCLUDED_COLUMNS & set(train.columns)))
    y_train = train[TARGET].astype(int)

    X_val = validation.drop(columns=list(EXCLUDED_COLUMNS & set(validation.columns)))
    y_val = validation[TARGET].astype(int)

    # Ensure train and validation use exactly the same feature columns.
    X_val = X_val.reindex(columns=X_train.columns)

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable model features were found.")

    print(f"Numerical features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")

    print("\n2/5 Building preprocessing pipeline...")

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    print("3/5 Training decision tree...")

    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=100,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Preprocessing is fitted only on the training data.
    model.fit(X_train, y_train)

    print("\n4/5 Evaluating on validation data...")

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)
    pr_auc = average_precision_score(y_val, probabilities)

    matrix = confusion_matrix(y_val, predictions, labels=[0, 1])

    report = [
        "OrderGuard — Decision Tree Baseline",
        "=" * 48,
        "Evaluation split: chronological validation set",
        "Class weighting: balanced",
        "Decision threshold: 0.50",
        "",
        f"Training rows: {len(X_train):,}",
        f"Validation rows: {len(X_val):,}",
        f"Features before encoding: {X_train.shape[1]}",
        "",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1-score:  {f1:.4f}",
        f"PR-AUC (average precision): {pr_auc:.4f}",
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        "                 Predicted on-time | Predicted late",
        f"Actual on-time:  {matrix[0, 0]:>16,} | {matrix[0, 1]:>14,}",
        f"Actual late:     {matrix[1, 0]:>16,} | {matrix[1, 1]:>14,}",
        "",
        "Classification report:",
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["on-time", "late"],
            zero_division=0,
        ),
        "",
        "Caution: results apply to delivered orders in this dataset.",
        "The delivery-estimate feature was excluded pending audit.",
    ]

    report_text = "\n".join(report)
    report_path = REPORT_DIR / "decision_tree_validation_metrics.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)

    print("\n5/5 Saving fitted pipeline...")

    model_path = ARTIFACT_DIR / "decision_tree_baseline.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {report_path}")
    print("\nBaseline training complete.")


if __name__ == "__main__":
    main()
