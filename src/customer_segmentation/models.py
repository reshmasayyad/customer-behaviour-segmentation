"""Customer-level holdout selection; no later-period data used for fitting."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .features import MODEL_FEATURES

SEED = 42


def log_features(features):
    values = features[MODEL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Features must be finite and non-negative")
    return np.log1p(values)


def clustering_metrics(values, labels):
    unique, counts = np.unique(labels, return_counts=True)
    valid = 1 < len(unique) < len(labels)
    return {
        "silhouette": float(silhouette_score(values, labels, sample_size=min(1500, len(labels)), random_state=SEED)) if valid else np.nan,
        "davies_bouldin": float(davies_bouldin_score(values, labels)) if valid else np.nan,
        "min_cluster_share": float(counts.min() / len(labels)),
        "observed_clusters": len(unique),
    }


def fit_kmeans_candidates(features, counts=range(2, 7)):
    """Select k using a customer holdout silhouette, then refit all early customers.

    Candidate holdout clusters must each contain at least 1% of holdout customers;
    this guards against a tiny cluster driving an unhelpful segmentation.
    """
    raw = log_features(features)
    train_idx, validation_idx = train_test_split(np.arange(len(raw)), test_size=0.3, random_state=SEED)
    scaler = StandardScaler().fit(raw[train_idx])
    train, valid = scaler.transform(raw[train_idx]), scaler.transform(raw[validation_idx])
    rows = []
    for k in counts:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(train)
        metrics = clustering_metrics(valid, model.predict(valid))
        rows.append({"model": "KMeans", "k": k, "train_inertia": model.inertia_, **metrics})
    candidates = pd.DataFrame(rows)
    eligible = candidates.loc[candidates.min_cluster_share.ge(0.01) & candidates.observed_clusters.eq(candidates.k)]
    if eligible.empty:
        raise RuntimeError("No candidate satisfies the minimum cluster-size rule")
    best = eligible.sort_values("silhouette", ascending=False).iloc[0]
    final_scaler = StandardScaler().fit(raw)
    final_values = final_scaler.transform(raw)
    final_model = KMeans(n_clusters=int(best.k), random_state=SEED, n_init=20).fit(final_values)
    return {"scaler": final_scaler, "model": final_model, "values": final_values, "candidates": candidates,
            "selected_k": int(best.k), "train_idx": train_idx, "validation_idx": validation_idx}


def transform(bundle, features):
    return bundle["scaler"].transform(log_features(features))


def predict(bundle, features):
    return bundle["model"].predict(transform(bundle, features))
