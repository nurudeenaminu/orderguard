from src.orderguard.models.train_champion import (
    FEATURES as TRAINING_FEATURES,
)
from src.orderguard.models.predict_orders import (
    FEATURES as INFERENCE_FEATURES,
)

EXPECTED_FEATURES = {
    "customer_state",
    "dominant_seller_state",
    "item_price_mean",
    "item_price_max",
    "seller_customer_same_state_fraction",
    "distinct_sellers",
    "purchase_month",
}

FORBIDDEN_FEATURES = {
    "order_id",
    "order_status",
    "is_late",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "promised_delivery_days",
    "payment_value_sum",
}


def test_training_and_inference_use_same_feature_contract():
    assert set(TRAINING_FEATURES) == EXPECTED_FEATURES
    assert set(INFERENCE_FEATURES) == EXPECTED_FEATURES
    assert len(TRAINING_FEATURES) == len(set(TRAINING_FEATURES))
    assert len(INFERENCE_FEATURES) == len(set(INFERENCE_FEATURES))


def test_forbidden_columns_are_not_model_features():
    assert not (set(TRAINING_FEATURES) & FORBIDDEN_FEATURES)
    assert not (set(INFERENCE_FEATURES) & FORBIDDEN_FEATURES)


def test_feature_contract_excludes_payment_value():
    assert "payment_value_sum" not in TRAINING_FEATURES
    assert "payment_value_sum" not in INFERENCE_FEATURES
