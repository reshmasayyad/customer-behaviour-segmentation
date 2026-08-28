import pandas as pd
from customer_segmentation.evaluation import temporal_comparison


def test_missing_activity_is_not_silently_dropped():
    early = pd.Series(["S1", "S1", "S2"], index=["a", "b", "c"], name="segment")
    later = pd.Series(["S2", "S2", "S1"], index=["a", "c", "d"], name="segment")
    transitions, cohort, summary = temporal_comparison(early, later)
    assert transitions.to_numpy().sum() == 2
    assert cohort.to_numpy().sum() == 3
    assert summary["no_purchase_in_later_window"] == 1
    assert summary["later_active_not_observed_in_early_window"] == 1
    assert summary["same_segment_share_among_active_in_both"] == 0.5
