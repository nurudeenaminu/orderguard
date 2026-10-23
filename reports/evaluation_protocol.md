# OrderGuard Evaluation Protocol

## 1. Purpose

This document defines how OrderGuard models should be selected, evaluated, and approved for further use.

The goal is to estimate whether the model can rank orders that will be delivered late above orders that will arrive on time.

## 2. Target Definition

The current target is:

- `is_late = 1`: actual customer delivery occurred after the estimated delivery date.
- `is_late = 0`: actual customer delivery occurred on or before the estimated delivery date.

Only orders with a delivered status and valid purchase, actual-delivery, and estimated-delivery timestamps are included.

The model does not currently predict cancellations, non-delivery, or the outcomes of orders still in transit.

Because the target is observed only for eligible completed deliveries, performance on this population should not automatically be generalized to every newly placed order.

## 3. Intended Prediction Moment

The working assumption is that an order is scored after checkout and once its item and seller information is finalized, but before fulfillment and delivery outcomes occur.

This assumption requires confirmation against the intended operational workflow.

All model features must be available at or before the prediction timestamp. The feature-availability audit remains incomplete.

The current champion uses seven features:

- `customer_state`
- `dominant_seller_state`
- `item_price_mean`
- `item_price_max`
- `seller_customer_same_state_fraction`
- `distinct_sellers`
- `purchase_month`

`payment_value_sum` is excluded from the current champion because chronological ablation found negligible ranking differences and its availability at prediction time remains unverified.

## 4. Existing Data Splits

The current dataset uses chronological splits:

| Split      |   Rows | Purchase-date period         |
| ---------- | -----: | ---------------------------- |
| Training   | 67,529 | September 2016 to April 2018 |
| Validation | 14,470 | April 2018 to June 2018      |
| Test       | 14,471 | June 2018 to August 2018     |

These periods are not independent of model development.

The validation period was used repeatedly for model and feature selection. The test period was also examined for performance evaluation and error analysis.

Therefore, the existing test results are historical development diagnostics, not an untouched final estimate of the current champion's generalization performance.

## 5. Model Selection

During development, model candidates should be compared using chronological validation rather than random splitting.

Primary ranking metric:

- Average precision (area under the precision-recall curve).

Supporting measures:

- Average precision relative to the validation period's late-order prevalence.
- Precision and recall at explicitly defined alert budgets.
- F1 score where a precision-recall balance is useful.
- Brier score, log loss, and calibration diagnostics when interpreting scores as probabilities.

Late-order prevalence varies over time, so average precision should be interpreted in the context of each evaluation period's prevalence.

Thresholds and alert budgets must be selected using development data, not the final holdout.

## 6. Independent Final Evaluation

A genuinely untouched evaluation period must be obtained before making strong claims about future performance.

Preferred approach:

1. Freeze the selected feature definitions, preprocessing, model configuration, and alert policy.
2. Obtain a later chronological period or an appropriate external dataset.
3. Apply the frozen feature-generation process without using evaluation outcomes.
4. Score the eligible orders.
5. Evaluate ranking, alert workload, precision, recall, and probability calibration where appropriate.
6. Report the evaluation period, late-order prevalence, sample size, and uncertainty.
7. Do not tune the model against this final evaluation. If tuning resumes, obtain a new independent holdout for subsequent final claims.

The current dataset does not provide an untouched holdout because all its periods have already been inspected during development.

## 7. Temporal and Data-Quality Risks

The following risks require continued attention:

- Late-order prevalence changes over time.
- Model performance varies across chronological periods.
- Geographic feature relationships may change.
- Some source fields may not be available at the intended prediction moment.
- Orders without observed delivery outcomes may be missing from the supervised target.
- The current probability estimates have not been shown to remain calibrated across time.

These risks must be considered when evaluating whether the model is suitable for operational use.

## 8. Reporting Requirements

Each evaluation report should record:

- Dataset version and evaluation period.
- Eligibility rules and late-order prevalence.
- Feature list and model configuration.
- Average precision and supporting ranking metrics.
- Precision, recall, and number of orders flagged at the chosen alert budget.
- Calibration metrics if probability interpretation is intended.
- Known data limitations and any deviations from the protocol.

## 9. Approval Status

OrderGuard remains a research and portfolio prototype.

The current champion has not been independently evaluated on an untouched future period, and point-in-time feature availability remains partially unresolved.

It is not approved for production deployment.
