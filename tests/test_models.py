import numpy as np
import pandas as pd
from customer_segmentation.features import MODEL_FEATURES
from customer_segmentation.models import fit_kmeans_candidates, fit_gmm_candidates, membership, predict


def synthetic_features():
    rng = np.random.default_rng(7)
    x = np.vstack([rng.normal([2, 1, 3, 1], 0.12, (60, 4)), rng.normal([5, 4, 7, 4], 0.12, (60, 4))])
    return pd.DataFrame(np.expm1(x), columns=MODEL_FEATURES)


def test_holdout_is_disjoint_and_predictions_repeat():
    frame = synthetic_features()
    bundle = fit_kmeans_candidates(frame, counts=[2, 3])
    assert not set(bundle["train_idx"]) & set(bundle["validation_idx"])
    assert bundle["selected_k"] == 2
    assert np.array_equal(predict(bundle, frame), predict(bundle, frame))


def test_mixture_membership_sums_to_one():
    frame = synthetic_features()
    bundle = fit_gmm_candidates(frame, counts=[2, 3], covariance_types=["diag"])
    probabilities = membership(bundle, frame)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities.min().min() >= 0
