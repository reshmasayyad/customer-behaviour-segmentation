"""Separate model robustness, customer movement and future purchasing."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from .models import SEED, log_features, predict


def segment_names(labels, features):
    """Stable display IDs ordered by median spending; names do not imply value truth."""
    spending = features.assign(component=labels).groupby("component").monetary_gbp.median().sort_values()
    return {int(component): f"S{rank + 1}" for rank, component in enumerate(spending.index)}


def label_series(bundle, features, names):
    return pd.Series([names[int(x)] for x in predict(bundle, features)], index=features.index, name="segment")


def bootstrap_stability(bundle, features, repetitions=20):
    """Refit scaler and selected model on bootstrap samples, compare common anchors.

    ARI is invariant to component-number permutations. Selected k and covariance
    form stay fixed; these results do not measure uncertainty in model selection.
    Bootstrap fitting also changes initialization seeds. The empirical interval
    is a robustness summary, not a population confidence interval.
    """
    rng = np.random.default_rng(SEED)
    raw = log_features(features)
    anchor = np.sort(rng.choice(len(raw), size=min(1500, len(raw)), replace=False))
    reference = bundle["model"].predict(bundle["scaler"].transform(raw[anchor]))
    rows = []
    for repetition in range(repetitions):
        indices = rng.choice(len(raw), size=len(raw), replace=True)
        scaler = StandardScaler().fit(raw[indices])
        model = clone(bundle["model"]).set_params(random_state=SEED + repetition + 1, n_init=3)
        model.fit(scaler.transform(raw[indices]))
        labels = model.predict(scaler.transform(raw[anchor]))
        rows.append({"repetition": repetition + 1, "ari": adjusted_rand_score(reference, labels),
                     "converged": bool(getattr(model, "converged_", True))})
    return pd.DataFrame(rows)


def segment_profiles(features, labels, outcomes):
    """Summarize interpretable features and 90-day observed repeat purchasing."""
    frame = features.join(labels).join(outcomes)
    profiles = frame.groupby("segment").agg(
        customers=("segment", "size"), median_recency_days=("recency_days", "median"),
        median_orders=("frequency_orders", "median"), median_spend_gbp=("monetary_gbp", "median"),
        median_products=("unique_products", "median"), median_order_value_gbp=("average_order_gbp", "median"),
        total_spend_gbp=("monetary_gbp", "sum"), repeat_purchase_rate_90d=("purchased_again", "mean"),
        repeat_customers_90d=("purchased_again", "sum"), mean_future_spend_gbp=("future_spend_gbp", "mean"),
    )
    profiles["customer_share"] = profiles.customers / profiles.customers.sum()
    profiles["spend_share"] = profiles.total_spend_gbp / profiles.total_spend_gbp.sum()
    n, p, z = profiles.customers, profiles.repeat_purchase_rate_90d, 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    profiles["repeat_rate_wilson_low"] = center-half
    profiles["repeat_rate_wilson_high"] = center+half
    return profiles


def temporal_comparison(early_labels, later_labels):
    """Movement with a frozen model; missing activity is reported separately."""
    shared = early_labels.index.intersection(later_labels.index)
    inactive = early_labels.index.difference(later_labels.index)
    newly_observed = later_labels.index.difference(early_labels.index)
    transitions = pd.crosstab(early_labels.loc[shared], later_labels.loc[shared],
                             rownames=["early_segment"], colnames=["later_segment"])
    all_destinations = later_labels.reindex(early_labels.index).fillna("No purchases in later window")
    cohort_table = pd.crosstab(early_labels, all_destinations,
                              rownames=["early_segment"], colnames=["later_activity"])
    summary = {
        "early_active_customers": len(early_labels), "later_active_customers": len(later_labels),
        "active_in_both": len(shared), "no_purchase_in_later_window": len(inactive),
        "later_active_not_observed_in_early_window": len(newly_observed),
        "same_segment_share_among_active_in_both": float((early_labels.loc[shared] == later_labels.loc[shared]).mean()),
    }
    return transitions, cohort_table, summary


def regularization_sensitivity(bundle, features, values=(0.0001, 0.001, 0.05)):
    """Test covariance floors with fixed k, feature space and model family.

    Count features can cause nearly point-mass Gaussian components; high mixture
    confidence is not evidence that business segments are well calibrated.
    """
    scaled = bundle["scaler"].transform(log_features(features))
    reference = bundle["model"].predict(scaled)
    rows = []
    for regularization in values:
        model = clone(bundle["model"]).set_params(reg_covar=regularization, random_state=SEED, n_init=5)
        model.fit(scaled)
        rows.append({"regularization": regularization,
                     "ari_to_primary": adjusted_rand_score(reference, model.predict(scaled)),
                     "mean_membership_confidence": model.predict_proba(scaled).max(axis=1).mean(),
                     "converged": bool(model.converged_)})
    return pd.DataFrame(rows)
