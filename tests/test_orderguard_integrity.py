import pytest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "orderguard_champion.joblib"

FEATURES = [
    "customer_state",
    "dominant_seller_state",
    "item_price_mean",
    "item_price_max",
    "seller_customer_same_state_fraction",
    "distinct_sellers",
    "purchase_month",
]


def load_split(name):
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        pytest.skip(f"Local dataset split is not available: {path}")

    return pd.read_csv(path)


def test_split_files_exist_and_have_expected_sizes():
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    assert len(train) == 67_529
    assert len(validation) == 14_470
    assert len(test) == 14_471


def test_order_ids_are_present_and_unique():
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        assert frame["order_id"].notna().all(), f"{name} contains missing order IDs"
        assert frame["order_id"].is_unique, f"{name} contains duplicate order IDs"

    train_ids = set(train["order_id"])
    validation_ids = set(validation["order_id"])
    test_ids = set(test["order_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_target_contains_only_binary_labels():
    for name in ["train", "validation", "test"]:
        frame = load_split(name)

        assert frame["is_late"].notna().all(), f"{name} contains missing targets"
        assert set(frame["is_late"].unique()).issubset(
            {0, 1}
        ), f"{name} contains labels other than 0 and 1"


def test_splits_are_chronological_and_non_overlapping():
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    for frame in [train, validation, test]:
        frame["order_purchase_timestamp"] = pd.to_datetime(
            frame["order_purchase_timestamp"],
            errors="raise",
        )

    assert (
        train["order_purchase_timestamp"].max()
        < validation["order_purchase_timestamp"].min()
    ), "Training data overlaps the validation period"

    assert (
        validation["order_purchase_timestamp"].max()
        < test["order_purchase_timestamp"].min()
    ), "Validation data overlaps the test period"


def test_model_artifact_can_score_orders():
    if not MODEL_PATH.exists():
        pytest.skip("Local champion model artifact is not available.")
    assert MODEL_PATH.exists(), f"Champion artifact is missing: {MODEL_PATH}"

    model = joblib.load(MODEL_PATH)
    validation = load_split("validation")

    missing = set(FEATURES) - set(validation.columns)
    assert not missing, f"Validation data lacks features: {sorted(missing)}"

    X = validation[FEATURES].head(32).copy()

    positive_class_index = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, positive_class_index]

    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_model_handles_a_missing_numeric_value():
    if not MODEL_PATH.exists():
        pytest.skip("Local champion model artifact is not available.")
    model = joblib.load(MODEL_PATH)
    validation = load_split("validation")

    X = validation[FEATURES].head(8).copy()

    # The trained pipeline should impute this missing numeric value.
    X.loc[X.index[0], "item_price_mean"] = np.nan

    positive_class_index = list(model.classes_).index(1)
    scores = model.predict_proba(X)[:, positive_class_index]

    assert len(scores) == len(X)
    assert np.isfinite(scores).all()
