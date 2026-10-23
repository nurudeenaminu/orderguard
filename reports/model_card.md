# OrderGuard Model Card

## Model Identity

**Model name:** OrderGuard Champion
**Model type:** Logistic regression
**Version:** Initial portfolio prototype
**Artifact:** `artifacts/orderguard_champion.joblib`
**Training script:** `src/orderguard/models/train_champion.py`

## Intended Purpose

OrderGuard ranks delivered-order records according to their relative risk of late delivery. The intended research use is to explore whether an e-commerce operations team could prioritize orders for review.

The model is not validated for autonomous operational decisions.

## Intended Prediction Target

The target is whether an order was delivered after its estimated delivery date.

The current target applies only to orders with a delivered status and valid purchase, actual-delivery, and estimated-delivery timestamps.

Cancellations, orders still in transit, and orders with insufficient delivery-date information are outside the current target definition.

## Training Data

**Dataset:** Public Olist Brazilian e-commerce data
**Eligible observations:** 96,470 orders

| Split      |   Rows | Late-order rate |
| ---------- | -----: | --------------: |
| Training   | 67,529 |           9.03% |
| Validation | 14,470 |           5.34% |
| Test       | 14,471 |           6.61% |

The splits are chronological. The changing late-order rates indicate that the delivery environment differs across time periods.

The original test period was used in multiple diagnostic and modeling discussions. It must not be treated as an untouched holdout for the current champion.

## Model Specification

The champion uses eight features:

- `customer_state`
- `dominant_seller_state`
- `item_price_mean`
- `item_price_max`
- `seller_customer_same_state_fraction`
- `distinct_sellers`
- `purchase_month`

The pipeline uses median imputation and standard scaling for numeric features, most-frequent imputation and one-hot encoding for categorical features, and logistic regression with no class weighting.

Training samples receive exponentially decaying recency weights with a 90-day half-life.

Feature availability at the prediction timestamp has not been fully verified. The training and inference feature-generation processes must be audited for temporal leakage before the model is considered deployable.

## Validation Performance

| Metric                   | Result |
| ------------------------ | -----: |
| Average precision        | 0.0909 |
| Brier score              | 0.0567 |
| Log loss                 | 0.2284 |
| Mean predicted score     | 0.1171 |
| Observed late-order rate |  5.34% |

At a 15% alert budget:

| Metric          | Result |
| --------------- | -----: |
| Orders flagged  |  2,171 |
| True positives  |    241 |
| False positives |  1,930 |
| False negatives |    532 |
| Precision       | 11.10% |
| Recall          | 31.18% |
| F1              | 0.1637 |

These are validation results used during development. They are not independent estimates of expected production performance.

## Interpretation of Scores

The model scores should be interpreted as ranking scores, not as calibrated probabilities of lateness.

The mean validation score exceeds the observed late-order rate. Probability calibration experiments did not establish a reliable calibration method that transferred from an earlier period to the validation period.

## Known Limitations and Risks

1. Late-delivery rates change over time.
2. Performance is weak at identifying all late orders without generating many false alarms.
3. Geographic features are important and their relationships with delivery outcomes may change.
4. The current target excludes cancellations and orders without complete delivery timestamps.
5. Point-in-time availability of all features requires verification.
6. The original test set has been used repeatedly and is no longer an independent final evaluation set.
7. No operational alert policy or ongoing drift-monitoring process has been validated.
8. payment_value_sum was excluded from the current champion after chronological ablation showed negligible performance differences. The availability of payment totals at the intended prediction time remains unverified.

## Recommended Next Steps

- Audit feature availability at the prediction timestamp.
- Review the feature-generation pipeline for temporal leakage.
- Establish a new chronological holdout that is not used for model or threshold selection.
- Evaluate discrimination, calibration, and alert workload on that holdout.
- Define alert costs and business capacity with operational stakeholders.
- Add automated data checks and model evaluation tests.
- Establish monitoring for changes in feature distributions, late-order prevalence, and model performance.

## Approval Status

**Research prototype only. Not approved for production deployment.**
