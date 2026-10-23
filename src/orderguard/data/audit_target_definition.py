from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

MONTHLY_REPORT = REPORT_DIR / "target_definition_audit_monthly.csv"
STATUS_REPORT = REPORT_DIR / "target_definition_audit_status.csv"

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

IN_FLIGHT_STATUSES = {
    "created",
    "approved",
    "invoicing",
    "invoiced",
    "processing",
    "shipped",
}


def find_orders_file():
    """Find the raw Olist orders file without assuming a path silently."""
    candidates = [
        RAW_DIR / "olist_orders_dataset.csv",
        *RAW_DIR.glob("*orders_dataset*.csv"),
    ]

    existing = list(dict.fromkeys(path for path in candidates if path.exists()))

    if len(existing) != 1:
        raise FileNotFoundError(
            "Expected exactly one raw orders dataset. "
            f"Found: {[str(path) for path in existing]}"
        )

    return existing[0]


def main():
    source_path = find_orders_file()

    print(f"Reading raw orders from:\n{source_path}")

    orders = pd.read_csv(source_path)

    required_columns = {
        "order_id",
        "order_status",
        *DATE_COLUMNS,
    }

    missing_columns = required_columns - set(orders.columns)

    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")

    for column in DATE_COLUMNS:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    orders["purchase_month"] = (
        orders["order_purchase_timestamp"].dt.to_period("M").astype("string")
    )

    # Reproduce the target eligibility rule used by the project:
    # delivered order + known actual date + known promised date.
    orders["target_eligible"] = (
        orders["order_status"].eq("delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    )

    late_indicator = (
        orders["order_delivered_customer_date"]
        > orders["order_estimated_delivery_date"]
    ).astype("int8")

    orders["is_late"] = late_indicator.where(orders["target_eligible"], pd.NA).astype(
        "Int64"
    )
    # A rough diagnostic cutoff based on the latest observed purchase
    # or actual delivery timestamp. It is not a verified snapshot date.
    observed_timestamps = pd.concat(
        [
            orders["order_purchase_timestamp"],
            orders["order_delivered_customer_date"],
        ]
    )

    observed_cutoff_proxy = observed_timestamps.max()

    orders["potentially_censored_in_flight"] = (
        orders["order_status"].isin(IN_FLIGHT_STATUSES)
        & orders["order_delivered_customer_date"].isna()
        & orders["order_estimated_delivery_date"].notna()
        & (orders["order_estimated_delivery_date"] < observed_cutoff_proxy)
    )

    print(f"\nTotal raw orders: {len(orders):,}")
    print(f"Unique order IDs: {orders['order_id'].nunique():,}")
    print(f"Observed cutoff proxy: {observed_cutoff_proxy}")
    print(
        "Orders eligible for late/on-time target: "
        f"{int(orders['target_eligible'].sum()):,}"
    )
    print("Eligible orders classified late: " f"{int(orders['is_late'].sum()):,}")
    print(
        "Orders in in-flight statuses with missing delivery dates "
        "and estimated deadlines before the cutoff proxy: "
        f"{int(orders['potentially_censored_in_flight'].sum()):,}"
    )

    # Overall order-status and target-eligibility summary.
    status_summary = (
        orders.groupby("order_status", dropna=False)
        .agg(
            orders=("order_id", "size"),
            missing_purchase_date=(
                "order_purchase_timestamp",
                lambda x: int(x.isna().sum()),
            ),
            missing_actual_delivery_date=(
                "order_delivered_customer_date",
                lambda x: int(x.isna().sum()),
            ),
            missing_estimated_delivery_date=(
                "order_estimated_delivery_date",
                lambda x: int(x.isna().sum()),
            ),
            eligible_for_target=("target_eligible", "sum"),
            late_orders=("is_late", "sum"),
            potentially_censored_in_flight=("potentially_censored_in_flight", "sum"),
        )
        .reset_index()
    )

    status_summary["share_of_all_orders"] = status_summary["orders"] / len(orders)

    # Month-by-month target eligibility and late-delivery rates.
    monthly_summary = (
        orders.groupby("purchase_month", dropna=False)
        .agg(
            total_orders=("order_id", "size"),
            delivered_status_orders=(
                "order_status",
                lambda x: int(x.eq("delivered").sum()),
            ),
            actual_delivery_dates_present=(
                "order_delivered_customer_date",
                lambda x: int(x.notna().sum()),
            ),
            eligible_orders=("target_eligible", "sum"),
            late_orders=("is_late", "sum"),
            potentially_censored_in_flight=("potentially_censored_in_flight", "sum"),
        )
        .reset_index()
    )

    monthly_summary["eligible_share_of_all_orders"] = (
        monthly_summary["eligible_orders"] / monthly_summary["total_orders"]
    )

    monthly_summary["late_rate_among_eligible"] = monthly_summary[
        "late_orders"
    ] / monthly_summary["eligible_orders"].replace(0, pd.NA)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    status_summary.to_csv(STATUS_REPORT, index=False)
    monthly_summary.to_csv(MONTHLY_REPORT, index=False)

    print("\nORDER STATUS SUMMARY")
    print(
        status_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nMONTHLY TARGET ELIGIBILITY AND LATE RATES")
    print(
        monthly_summary[
            [
                "purchase_month",
                "total_orders",
                "delivered_status_orders",
                "eligible_orders",
                "eligible_share_of_all_orders",
                "late_orders",
                "late_rate_among_eligible",
                "potentially_censored_in_flight",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nStatus report saved to: {STATUS_REPORT}")
    print(f"Monthly report saved to: {MONTHLY_REPORT}")

    print(
        "\nCaution: the cutoff is only a proxy. These diagnostics "
        "do not prove that an individual order was censored or late."
    )


if __name__ == "__main__":
    main()
