# OrderGuard Feature Availability Audit

## Intended Prediction Moment

Working assumption: score an order after checkout, once order contents are finalized, but before fulfillment and delivery begin.

This timing must be confirmed against the eventual operational workflow.

## Feature Review

| Feature                             | Source                                           | Status                         | Required verification                                                                                                         |
| ----------------------------------- | ------------------------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| purchase_month                      | Order purchase timestamp                         | Derivation verified            | Confirm timestamp is available when scoring occurs                                                                            |
| customer_state                      | Customer table joined through customer_id        | Conditional                    | Confirm state represents the order-associated address at prediction time                                                      |
| item_price_mean                     | Order-item price aggregation                     | Conditional                    | Confirm item lines and prices are finalized by prediction time                                                                |
| item_price_max                      | Order-item price aggregation                     | Conditional                    | Confirm item lines and prices are finalized by prediction time                                                                |
| distinct_sellers                    | Distinct seller IDs per order                    | Conditional                    | Confirm all order-item seller assignments are known                                                                           |
| dominant_seller_state               | Mode of seller states across order items         | Conditional                    | Confirm all order-item seller assignments are known                                                                           |
| seller_customer_same_state_fraction | Derived from seller and customer states per item | Conditional                    | Confirm all dependencies are known at scoring time                                                                            |
| payment_value_sum                   | Payment value aggregation                        | Excluded from current champion | Timing remains unverified; establish when payment records and final amounts become available before reconsidering the feature |
|  |

## Target Definition

The target is whether an order was delivered after its estimated delivery date.

Only orders marked delivered with valid purchase, actual-delivery, and estimated-delivery timestamps are included in the current supervised dataset.

Orders that are canceled, still in transit, or missing required delivery timestamps are not labeled as on-time or late.

This creates a conditional target population: completed deliveries with known outcomes. The relationship between this population and orders scored before their outcomes are known must be evaluated before production use.

## Leakage Assessment

The eight selected features do not directly use the actual delivery date or estimated delivery date in the feature definitions shown.

However, availability at prediction time has not been conclusively verified for all source-table fields. The audit is therefore incomplete, and the current model is not approved for production deployment.

## Required Next Actions

1. Confirm the operational prediction timestamp.
2. Verify availability of order-item, customer, seller, and payment data at that timestamp.
3. Run a temporal feature-ablation experiment without payment_value_sum.
4. Review the current target population against the intended use case.
5. Re-evaluate on a genuinely untouched chronological holdout after the feature definition is frozen.

## Status

Initial static-code review completed. Point-in-time availability is only partially verified.
