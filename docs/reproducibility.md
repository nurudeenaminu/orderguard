# OrderGuard Reproducibility Guide

This guide explains how to set up OrderGuard from a fresh Git clone, prepare its data, train the champion model, and run its tests.

## 1. Prerequisites

- Git
- Python 3.11
- PowerShell on Windows, or an equivalent terminal
- Access to the public Olist dataset

## 2. Clone the Repository

```powershell
git clone https://github.com/nurudeenaminu/orderguard.git
cd orderguard
```

## 3. Create the Python Environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Verify the environment:

```powershell
python --version
python -m pip check
```

The project dependencies are pinned in `requirements.txt`. These pins cover the declared direct dependencies, not every transitive dependency.

## 4. Obtain the Dataset

Download the Brazilian E-Commerce Public Dataset by Olist from:

https://www.kaggle.com/olistbr/brazilian-ecommerce

Review the dataset's license and terms before using or redistributing its contents.

Extract the dataset CSV files into:

```text
data/raw/
```

Keep the original CSV filenames unchanged. The dataset files are intentionally excluded from Git, so each developer must obtain their own local copy.

Before continuing, inspect the available files:

```powershell
Get-ChildItem .\data\raw\*.csv
```

The data preparation script determines which source tables are required. If it reports a missing source CSV, confirm that the corresponding file from the Olist download is present in `data/raw/`.

## 5. Inspect the Source Data

Run:

```powershell
python .\src\orderguard\data\inspect_data.py
```

This script inspects the raw dataset and generates data inventory and column-profile reports.

Generated CSV reports are local artifacts and are not committed to Git.

## 6. Build the Modeling Dataset

Run:

```powershell
python .\src\orderguard\data\build_dataset.py
```

The script prepares the modeling dataset and writes:

```text
data/processed/orderguard_dataset.csv
```

The current target predicts lateness among delivered orders with valid purchase, actual-delivery, and estimated-delivery timestamps.

It does not predict cancellations or the outcomes of orders that are still in transit.

## 7. Create Chronological Splits

Run:

```powershell
python .\src\orderguard\data\split_dataset.py
```

The script creates chronological training, validation, and test files in:

```text
data/processed/
```

Expected row counts for the current dataset version are:

| Split      |   Rows |
| ---------- | -----: |
| Training   | 67,529 |
| Validation | 14,470 |
| Test       | 14,471 |

If your data differs from these counts, investigate the dataset version and preparation process rather than changing test assertions without explanation.

## 8. Train the Champion Model

Run:

```powershell
python .\src\orderguard\models\train_champion.py
```

The current champion is logistic regression trained with:

- Seven selected features
- No class weighting
- Recency sample weights with a 90-day half-life
- Median imputation and standard scaling for numeric features
- Most-frequent imputation and one-hot encoding for categorical features

The model artifact is saved locally to:

```text
artifacts/orderguard_champion.joblib
```

Validation metrics, alert-budget reports, and predictions are written to `reports/`.

The model artifact and generated reports are intentionally excluded from Git. Rebuild the artifact locally by running the training script.

## 9. Run the Tests

Run:

```powershell
python -m pytest .\tests -v
```

Some integrity tests depend on local processed data and the trained artifact. If those files are unavailable, these tests skip themselves. Portable feature-contract and synthetic-data tests should still run.

To exercise the complete local test suite, complete the data-preparation and model-training steps first.

## 10. Score Orders

After training the champion, score a CSV containing the seven required model features:

```powershell
python .\src\orderguard\models\predict_orders.py `
  .\data\processed\validation.csv `
  --output .\reports\scored_validation.csv `
  --alert-share 0.15
```

The inference script produces model scores, risk ranks, and alert flags.

The alert share is a configurable capacity scenario, not a validated production policy. The scores have not been established as calibrated probabilities of late delivery.

## 11. Git and Local Files

The repository intentionally excludes:

- `.venv/`
- Raw and processed datasets
- Saved model artifacts
- Generated CSV and text reports
- Local test caches

The relevant `.gitkeep` files preserve otherwise-empty directories in Git.

## 12. Known Limitations

- The current model is a research and portfolio prototype, not approved for production.
- Feature availability at the intended prediction time is not fully verified.
- The target only covers eligible completed deliveries.
- Model performance changes across chronological periods.
- The original test set was examined during development and is no longer an untouched final holdout.
- Probability calibration and the operational alert policy require further evaluation.

Before making production-readiness claims, establish point-in-time feature correctness and evaluate a frozen model on a genuinely untouched chronological or external dataset.
