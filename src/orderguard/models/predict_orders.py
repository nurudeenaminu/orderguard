from pathlib import Path
import argparse
import math

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "orderguard_champion.joblib"

FEATURES = [
    "customer_state",
    "dominant_seller_state",
    "item_price_mean",
    "item_price_max",
    "seller_customer_same_state_fraction",
    "distinct_sellers",
    "purchase_month",
]

METADATA_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score orders with the trained OrderGuard model. "
            "Scores rank orders by relative late-delivery risk; "
            "they are not calibrated probabilities."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="CSV containing the required eight model features.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "scored_orders.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved champion model.",
    )

    parser.add_argument(
        "--alert-share",
        type=float,
        default=None,
        help=(
            "Optional fraction of orders to flag, e.g. 0.15 "
            "for the highest-scoring 15 percent."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    if args.alert_share is not None and not 0 < args.alert_share <= 1:
        raise ValueError("--alert-share must be greater than 0 and at most 1.")

    print(f"Loading model: {args.model}")
    model = joblib.load(args.model)

    print(f"Reading input data: {args.input}")
    data = pd.read_csv(args.input)

    if data.empty:
        raise ValueError("The input CSV contains no orders.")

    missing = set(FEATURES) - set(data.columns)

    if missing:
        raise ValueError(
            "Input CSV is missing required feature columns: " f"{sorted(missing)}"
        )

    # Preserve identifying fields, but never export an existing target.
    available_metadata = [
        column for column in METADATA_COLUMNS if column in data.columns
    ]

    X = data[FEATURES].copy()

    # Treat empty strings as missing values so that the model's
    # imputers can handle them.
    for column in X.columns:
        if pd.api.types.is_object_dtype(X[column]) or pd.api.types.is_string_dtype(
            X[column]
        ):
            X[column] = X[column].replace(r"^\s*$", np.nan, regex=True)

    positive_class_index = list(model.classes_).index(1)

    print(f"Scoring {len(data):,} orders...")

    numeric_values = X.select_dtypes(include=np.number).to_numpy(dtype=float)

    if np.isinf(numeric_values).any():
        raise ValueError(
            "Numeric input contains infinite values. " "Please clean the input data."
        )

    scores = model.predict_proba(X)[:, positive_class_index]

    if not np.isfinite(scores).all():
        raise ValueError("The model produced non-finite scores.")

    output = data[available_metadata].copy()

    output["model_score"] = scores

    # Rank 1 is the order with the highest score.
    output["risk_rank"] = (
        pd.Series(scores).rank(method="first", ascending=False).astype(int).to_numpy()
    )

    if args.alert_share is not None:
        number_to_flag = min(
            len(output),
            math.ceil(len(output) * args.alert_share),
        )

        output["alert_flag"] = output["risk_rank"] <= number_to_flag

    output = output.sort_values(
        "risk_rank",
        ascending=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print("\nSCORING SUMMARY")
    print(f"Orders scored: {len(output):,}")
    print(f"Highest model score: {scores.max():.4f}")
    print(f"Lowest model score: {scores.min():.4f}")

    if args.alert_share is not None:
        print(f"Alert budget: {args.alert_share:.1%}")
        print(f"Orders flagged: {number_to_flag:,}")

    print(f"\nScored output saved to: {args.output}")


if __name__ == "__main__":
    main()
