# OrderGuard — Late Delivery Risk Ranking

OrderGuard is a classical machine learning project that ranks e-commerce orders by their relative risk of late delivery. It uses chronological training and validation, feature engineering, model comparison, recency weighting, and alert-budget analysis.

The project uses the public Olist Brazilian e-commerce dataset.

> **Project status:** Research and portfolio prototype. The model has not been independently validated for production use.

## Problem Definition

The target is binary:

- `is_late = 1`: the order was delivered after its estimated delivery date.
- `is_late = 0`: the order was delivered on or before its estimated delivery date.

The current dataset contains 96,470 eligible delivered orders, including 7,826 late orders.

This project predicts lateness among orders that were ultimately delivered and had the necessary timestamps. It does not currently predict cancellations, non-delivery, or whether an order still in transit will eventually be late.

## Current Model

The current champion is a logistic regression pipeline with:

- Eight selected input features.
- Median imputation and standard scaling for numeric features.
- Most-frequent imputation and one-hot encoding for categorical features.
- No class weighting.
- Exponentially decaying training sample weights with a 90-day half-life.

Older training examples receive less influence than recent examples. The half-life controls the decay rate; it does not discard older orders entirely.

### Features

1. `customer_state`
2. `dominant_seller_state`
3. `item_price_mean`
4. `item_price_max`
5. `seller_customer_same_state_fraction`
6. `distinct_sellers`
7. `purchase_month`

Feature availability at the intended prediction time must be verified before production use. The feature-generation process must also be consistent between training and inference.

The estimated delivery-date feature was excluded from earlier experiments because its availability at the time of prediction had not been verified.

## Data Splitting

Orders were split chronologically to reduce the risk of training on future orders and evaluating on older ones.

| Split      | Orders |
| ---------- | -----: |
| Training   | 67,529 |
| Validation | 14,470 |
| Test       | 14,471 |

The late-order rate is not constant over time. The training rate is approximately 9.03%, compared with 5.34% in validation and 6.61% in the test period.

Several experiments used the original validation and test periods for model selection and error analysis. Therefore, the existing test results are not an untouched independent estimate of the final champion's performance.

## Validation Results

The latest champion was trained on the full training split and evaluated on the validation split.

| Metric                            | Result |
| --------------------------------- | -----: |
| Average precision                 | 0.0916 |
| Brier score                       | 0.0569 |
| Log loss                          | 0.2287 |
| Mean predicted score              | 0.1174 |
| Actual validation late-order rate |  5.34% |

These results are development diagnostics, not independent estimates of future performance.

The mean predicted score is higher than the observed late-order rate. The scores should not be interpreted as calibrated probabilities of late delivery.

## Alert-Budget Analysis

Orders can be ranked by model score, and a chosen proportion of the highest-scoring orders can be flagged for review.

For a 15% alert budget on the validation set:

| Metric          | Result |
| --------------- | -----: |
| Orders flagged  |  2,171 |
| True positives  |    241 |
| False positives |  1,930 |
| False negatives |    532 |
| Precision       | 11.10% |
| Recall          | 31.18% |
| F1              | 0.1637 |

The alert budget is an operational scenario, not a validated production threshold. Selecting the top 15% requires a defined batch or review window. A fixed percentage does not mean a fixed risk level across time.

## Project Structure

```text
orderguard/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── orderguard/
│       ├── data/
│       └── models/
├── artifacts/
├── reports/
├── tests/
└── README.md
```

- `data/raw/`: original datasets.
- `data/processed/`: prepared dataset and chronological splits.
- `src/orderguard/data/`: dataset preparation, splitting, and data audits.
- `src/orderguard/models/`: training, evaluation, and inference scripts.
- `artifacts/`: saved model pipelines.
- `reports/`: model metrics, diagnostics, and predictions.
- `notebooks/`: exploratory analysis and experiments.
- `tests/`: automated tests.

## Setup

Use Python 3.11 or another environment compatible with the installed project dependencies.

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install numpy pandas scikit-learn joblib
```

If the project has a `requirements.txt`, install from that file instead:

```powershell
python -m pip install -r requirements.txt
```

## Train the Champion

From the project root:

```powershell
python .\src\orderguard\models\train_champion.py
```

This generates:

- `artifacts/orderguard_champion.joblib`
- `reports/champion_validation_metrics.txt`
- `reports/champion_validation_alert_budgets.csv`
- `reports/champion_validation_predictions.csv`

## Score Orders

The input CSV must contain the eight required features listed above.

```powershell
python .\src\orderguard\models\predict_orders.py `
  .\data\processed\validation.csv `
  --output .\reports\smoke_test_scored_orders.csv `
  --alert-share 0.15
```

The inference script writes model scores and risk ranks without exporting the existing target column.

The scoring pipeline expects the same feature definitions and compatible data types used during training.

## Known Limitations

- The current model has weak precision and misses many late orders.
- Prediction performance changes across chronological validation periods.
- Geographic patterns contribute substantial predictive signal and may change over time.
- Predicted scores are not established as calibrated risk probabilities.
- The target excludes cancellations and orders without sufficient delivery timestamps.
- Feature availability at prediction time still requires verification.
- The original test period has been examined repeatedly and is no longer an untouched final holdout.
- No production monitoring, retraining schedule, or operational alert policy has been validated.

## Responsible Use

OrderGuard should be used for experimentation and investigation, not as the sole basis for customer penalties, refunds, seller sanctions, or other consequential decisions.

Before production use, establish point-in-time feature correctness, evaluate performance on a genuinely untouched chronological holdout, assess probability calibration, define operational costs, and implement drift monitoring.
