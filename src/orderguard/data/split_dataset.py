from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "data" / "processed" / "orderguard_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"


def main():
    print("Loading modeling dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["order_purchase_timestamp"],
    )

    if df["order_purchase_timestamp"].isna().any():
        raise ValueError("Purchase timestamps contain missing values.")

    # Sort orders from earliest to latest before splitting.
    df = df.sort_values(
        "order_purchase_timestamp",
        kind="stable",
    ).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    print("\nChronological split summary")
    print("=" * 65)

    for name, split in splits.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        split.to_csv(output_path, index=False)

        late_rate = split["is_late"].mean()

        print(f"\n{name.upper()}")
        print(f"Rows: {len(split):,}")
        print(
            "Date range:",
            split["order_purchase_timestamp"].min(),
            "to",
            split["order_purchase_timestamp"].max(),
        )
        print(f"Late orders: {split['is_late'].sum():,}")
        print(f"Late rate: {late_rate:.2%}")
        print(f"Saved: {output_path}")

    # Confirm that the chronological periods do not overlap.
    assert (
        train["order_purchase_timestamp"].max()
        <= validation["order_purchase_timestamp"].min()
    )
    assert (
        validation["order_purchase_timestamp"].max()
        <= test["order_purchase_timestamp"].min()
    )

    print("\nChronological split checks passed.")


if __name__ == "__main__":
    main()
