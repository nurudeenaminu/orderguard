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


def test_champion_pipeline_trains_and_scores_synthetic_data():
    import numpy as np
    import pandas as pd

    from src.orderguard.models.train_champion import build_pipeline

    rows = []

    for i in range(40):
        rows.append(
            {
                "customer_state": ["SP", "RJ", "MG", "RS"][i % 4],
                "dominant_seller_state": ["SP", "RJ", "SC"][i % 3],
                "item_price_mean": float(20 + i * 3),
                "item_price_max": float(30 + i * 4),
                "seller_customer_same_state_fraction": float(i % 2),
                "distinct_sellers": 1 + (i % 3),
                "purchase_month": 1 + (i % 12),
            }
        )

    X = pd.DataFrame(rows)

    # Both classes are present. This is synthetic data for a
    # pipeline smoke test, not a model-quality experiment.
    y = np.array([int(i % 4 == 0 or i % 7 == 0) for i in range(len(X))])

    model = build_pipeline(X)
    model.fit(X, y)

    inference_rows = X.iloc[[0, 1]].copy()

    # Verify the preprocessing pipeline can handle missing numeric
    # values and previously unseen categorical values.
    inference_rows.loc[inference_rows.index[0], "item_price_mean"] = np.nan

    inference_rows.loc[inference_rows.index[1], "customer_state"] = "UNSEEN_STATE"

    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(inference_rows)[:, positive_index]

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
