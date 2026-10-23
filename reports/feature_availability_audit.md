# OrderGuard Feature Availability Audit

## Intended Prediction Moment

Working assumption: score an order after checkout, once order contents are finalized, but before fulfillment and delivery begin.

This timing must be confirmed against the eventual operational workflow.

## Feature Availability Review

Intended prediction moment: after checkout and after order contents are finalized, but before fulfillment and delivery outcomes occur.

| Feature                             | Source / derivation                                              | Status                         | Remaining requirement                                                                               |
| ----------------------------------- | ---------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------- |
| purchase_month                      | Month extracted from order_purchase_timestamp                    | Derivation verified            | Confirm the timestamp is available when scoring occurs                                              |
| customer_state                      | Customer table joined through customer_id                        | Conditional                    | Confirm this is the address associated with the order at prediction time                            |
| item_price_mean                     | Mean of price from order-item rows                               | Conditional                    | Confirm order items and prices are finalized before scoring                                         |
| item_price_max                      | Maximum price from order-item rows                               | Conditional                    | Confirm order items and prices are finalized before scoring                                         |
| distinct_sellers                    | Number of distinct seller IDs per order                          | Conditional                    | Confirm the complete order-item list is available                                                   |
| dominant_seller_state               | Most frequent seller state across order items                    | Conditional                    | Confirm seller assignments and relevant seller data are available                                   |
| seller_customer_same_state_fraction | Fraction of order items with matching customer and seller states | Conditional                    | Confirm all source values are available before fulfillment                                          |
| payment_value_sum                   | Payment records aggregated by order                              | Excluded from current champion | Timing remains unverified; reconsider only if availability is established and value is demonstrated |

## Audit Conclusion

The static dataset builder confirms how the selected features are constructed. It does not, by itself, prove when every source value becomes available.

The purchase-month derivation is consistent with the intended prediction time. The customer, seller, and order-item features remain conditional until their availability is confirmed against the operational data contract.

payment_value_sum is excluded from the current champion because chronological ablation showed negligible ranking differences and its point-in-time availability remains unverified.

The current model is a research prototype, not a production-approved system. Do not describe the feature-availability audit as fully passed.

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
