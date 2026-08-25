"""Features use half-open observation windows and no future purchases."""
import numpy as np
import pandas as pd

MODEL_FEATURES = ["recency_days", "frequency_orders", "monetary_gbp", "unique_products"]
EARLY_WINDOW = ("2009-12-01", "2010-06-01")
LATER_WINDOW = ("2010-12-01", "2011-06-01")


def customer_features(purchases, start, end):
    """One row per active customer in [start, end); recency is measured at end.

    Frequency counts DISTINCT invoices, not transaction lines. Monetary is
    gross positive merchandise spending in GBP. Inactive customers are not
    imputed as active; their absence is reported in the temporal analysis.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end:
        raise ValueError("start must precede end")
    window = purchases.loc[purchases.InvoiceDate.ge(start) & purchases.InvoiceDate.lt(end)].copy()
    if window.empty:
        raise ValueError("No purchases in the requested observation window")
    features = window.groupby("customer_id").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        frequency_orders=("invoice_id", "nunique"),
        monetary_gbp=("revenue", "sum"),
        unique_products=("stock_code", "nunique"),
        units=("Quantity", "sum"),
    )
    features["recency_days"] = (end.normalize() - features.last_purchase.dt.normalize()).dt.days
    features["average_order_gbp"] = features.monetary_gbp / features.frequency_orders
    features["observed_span_days"] = (features.last_purchase - features.first_purchase).dt.total_seconds() / 86400
    features["window_days"] = (end - start).days
    if not np.isfinite(features[MODEL_FEATURES]).all().all():
        raise ValueError("Non-finite customer features")
    return features.sort_index()


def future_outcomes(purchases, customer_ids, cutoff, days=90):
    """Join subsequent purchases only AFTER customer features are fixed."""
    cutoff = pd.Timestamp(cutoff)
    stop = cutoff + pd.Timedelta(days=days)
    available_end = purchases.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    if stop > available_end:
        raise ValueError("Follow-up extends beyond available dataset coverage")
    window = purchases.loc[purchases.InvoiceDate.ge(cutoff) & purchases.InvoiceDate.lt(stop)]
    observed = window.groupby("customer_id").agg(
        future_orders=("invoice_id", "nunique"), future_spend_gbp=("revenue", "sum")
    )
    outcomes = observed.reindex(customer_ids).fillna(0)
    outcomes["purchased_again"] = outcomes.future_orders.gt(0).astype(int)
    return outcomes
