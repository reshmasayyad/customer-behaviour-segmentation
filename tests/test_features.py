import pandas as pd
from customer_segmentation.features import customer_features, future_outcomes


def fixture_purchases():
    return pd.DataFrame({
        "customer_id": ["a", "a", "a", "b"],
        "invoice_id": ["1", "1", "2", "3"],
        "InvoiceDate": pd.to_datetime(["2010-01-01 12:00", "2010-01-01 12:00", "2010-02-01 00:00", "2010-01-10 00:00"]),
        "revenue": [10, 20, 999, 5], "stock_code": ["10001", "10002", "10003", "10001"],
        "Quantity": [1, 2, 99, 1],
    })


def test_distinct_orders_and_exclusive_cutoff_prevent_future_leakage():
    frame = fixture_purchases()
    result = customer_features(frame, "2010-01-01", "2010-02-01")
    assert result.loc["a", "frequency_orders"] == 1
    assert result.loc["a", "monetary_gbp"] == 30
    assert result.loc["a", "unique_products"] == 2
    assert result.loc["a", "recency_days"] == 31


def test_future_outcomes_include_non_returning_customers():
    result = future_outcomes(fixture_purchases(), ["a", "b", "c"], "2010-02-01", days=1)
    assert result.loc["a", "future_spend_gbp"] == 999
    assert result.loc["b", "purchased_again"] == 0
    assert result.loc["c", "future_orders"] == 0
