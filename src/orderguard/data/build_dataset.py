from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def read_csv(name, **kwargs):
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path, **kwargs)


def mode_or_nan(series):
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else None


def main():
    print("1/6 Loading Olist data...")

    orders = read_csv(
        "olist_orders_dataset.csv",
        usecols=[
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    customers = read_csv(
        "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"],
    )

    items = read_csv(
        "olist_order_items_dataset.csv",
        usecols=[
            "order_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ],
    )

    products = read_csv(
        "olist_products_dataset.csv",
        usecols=[
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
    )

    sellers = read_csv(
        "olist_sellers_dataset.csv",
        usecols=["seller_id", "seller_state"],
    )

    payments = read_csv(
        "olist_order_payments_dataset.csv",
        usecols=[
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_value",
            "payment_installments",
        ],
    )

    for column in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    print("2/6 Creating the prediction target...")

    # Restrict the first version to delivered orders with valid timestamps.
    # This predicts late delivery among delivered orders, not cancellations.
    orders = orders.loc[
        orders["order_status"].eq("delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    orders["is_late"] = (
        orders["order_delivered_customer_date"]
        > orders["order_estimated_delivery_date"]
    ).astype("int8")

    # Candidate order-time features. Do not use the estimated date itself as
    # a model feature until we establish when that estimate was recorded.
    purchase = orders["order_purchase_timestamp"]
    base = orders[["order_id", "order_purchase_timestamp", "is_late"]].copy()

    base["purchase_hour"] = purchase.dt.hour
    base["purchase_day_of_week"] = purchase.dt.dayofweek
    base["purchase_month"] = purchase.dt.month
    base["purchase_is_weekend"] = (purchase.dt.dayofweek >= 5).astype("int8")

    # Candidate only: validate that the original delivery estimate was
    # available at prediction time before using this feature.
    base["promised_delivery_days"] = (
        orders["order_estimated_delivery_date"] - purchase.dt.normalize()
    ).dt.total_seconds() / 86400

    base = base.merge(
        orders[["order_id", "customer_id"]].merge(
            customers, on="customer_id", how="left", validate="many_to_one"
        )[["order_id", "customer_state"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    print("3/6 Enriching and aggregating order items...")

    items = items.merge(products, on="product_id", how="left", validate="many_to_one")
    items = items.merge(sellers, on="seller_id", how="left", validate="many_to_one")

    translation_path = RAW / "product_category_name_translation.csv"
    if translation_path.exists():
        translations = pd.read_csv(translation_path)
        if {
            "product_category_name",
            "product_category_name_english",
        }.issubset(translations.columns):
            items = items.merge(
                translations,
                on="product_category_name",
                how="left",
                validate="many_to_one",
            )
            items["product_category"] = items["product_category_name_english"].fillna(
                items["product_category_name"]
            )
        else:
            items["product_category"] = items["product_category_name"]
    else:
        items["product_category"] = items["product_category_name"]

    items["seller_customer_same_state"] = (
        items["seller_state"]
        == items["order_id"].map(base.set_index("order_id")["customer_state"])
    ).astype("float")

    item_features = (
        items.groupby("order_id")
        .agg(
            item_count=("product_id", "size"),
            distinct_products=("product_id", "nunique"),
            distinct_sellers=("seller_id", "nunique"),
            item_price_sum=("price", "sum"),
            item_price_mean=("price", "mean"),
            item_price_max=("price", "max"),
            freight_sum=("freight_value", "sum"),
            freight_mean=("freight_value", "mean"),
            product_weight_mean=("product_weight_g", "mean"),
            product_length_mean=("product_length_cm", "mean"),
            product_height_mean=("product_height_cm", "mean"),
            product_width_mean=("product_width_cm", "mean"),
            dominant_category=("product_category", mode_or_nan),
            dominant_seller_state=("seller_state", mode_or_nan),
            seller_customer_same_state_fraction=("seller_customer_same_state", "mean"),
        )
        .reset_index()
    )

    item_features["freight_to_price_ratio"] = item_features[
        "freight_sum"
    ] / item_features["item_price_sum"].replace(0, float("nan"))

    print("4/6 Aggregating payment features...")

    payment_features = (
        payments.groupby("order_id")
        .agg(
            payment_count=("payment_sequential", "count"),
            payment_value_sum=("payment_value", "sum"),
            payment_installments_max=("payment_installments", "max"),
            dominant_payment_type=("payment_type", mode_or_nan),
        )
        .reset_index()
    )

    print("5/6 Combining features and checking integrity...")

    dataset = base.merge(
        item_features, on="order_id", how="left", validate="one_to_one"
    ).merge(payment_features, on="order_id", how="left", validate="one_to_one")

    if dataset["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found in the modeling dataset.")

    if not dataset["is_late"].isin([0, 1]).all():
        raise ValueError("Target contains values other than 0 and 1.")

    # Traceability columns are retained for future chronological splitting,
    # but must not be passed directly into the model.
    output_path = PROCESSED / "orderguard_dataset.csv"
    dataset.to_csv(output_path, index=False)

    missingness = pd.DataFrame(
        {
            "column": dataset.columns,
            "missing_count": dataset.isna().sum().values,
            "missing_percentage": (dataset.isna().mean().values * 100).round(2),
        }
    ).sort_values("missing_percentage", ascending=False)

    missingness.to_csv(REPORTS / "modeling_dataset_missingness.csv", index=False)

    summary = [
        "OrderGuard dataset summary",
        "==========================",
        f"Rows: {len(dataset):,}",
        f"Columns: {len(dataset.columns):,}",
        f"Unique order IDs: {dataset['order_id'].nunique():,}",
        f"Duplicate order IDs: {dataset['order_id'].duplicated().sum():,}",
        "",
        "Target counts:",
        dataset["is_late"].value_counts().sort_index().to_string(),
        "",
        "Target proportions:",
        dataset["is_late"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_string(),
        "",
        f"Dataset file: {output_path}",
        "Caution: validate feature availability time before training.",
    ]

    (REPORTS / "dataset_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("6/6 Dataset build complete.")
    print(f"Saved dataset: {output_path}")
    print(f"Saved summary: {REPORTS / 'dataset_summary.txt'}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Duplicate order IDs: {dataset['order_id'].duplicated().sum()}")


if __name__ == "__main__":
    main()
