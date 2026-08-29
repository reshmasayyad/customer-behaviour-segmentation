"""Save aggregate results, a reusable fitted model, and presentation figures."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .models import clustering_metrics, membership
from .features import MODEL_FEATURES, EARLY_WINDOW, LATER_WINDOW

COLORS = ["#2563eb", "#0d9488", "#d97706", "#9333ea", "#e11d48", "#475569"]


def save_results(root, early, later, km, gmm, early_labels, later_labels,
                 profiles, later_profiles, bootstrap, transitions, cohort, temporal):
    root = Path(root)
    out = root / "results"
    out.mkdir(exist_ok=True)
    km["candidates"].to_csv(out / "kmeans_selection.csv", index=False)
    gmm["candidates"].to_csv(out / "gmm_selection.csv", index=False)
    profiles.to_csv(out / "segment_profiles.csv")
    later_profiles.to_csv(out / "later_segment_profiles.csv")
    transitions.to_csv(out / "active_customer_transitions.csv")
    cohort.to_csv(out / "early_customer_later_activity.csv")
    stability_summary = []
    for name, table in bootstrap.items():
        table.to_csv(out / f"{name.lower()}_bootstrap.csv", index=False)
        stability_summary.append({"model": name, "median_ari": table.ari.median(),
                                  "empirical_5pct_ari": table.ari.quantile(.05),
                                  "empirical_95pct_ari": table.ari.quantile(.95),
                                  "repetitions": len(table), "all_converged": bool(table.converged.all())})
    pd.DataFrame(stability_summary).to_csv(out / "stability_summary.csv", index=False)
    comparisons = []
    for name, bundle in [("KMeans", km), ("GMM", gmm)]:
        comparisons.append({"model": name, "k": bundle["selected_k"],
                            **clustering_metrics(bundle["values"], bundle["model"].predict(bundle["values"]))})
    pd.DataFrame(comparisons).to_csv(out / "final_model_comparison.csv", index=False)
    probabilities = membership(gmm, early)
    summary = {"early_window": EARLY_WINDOW, "later_window": LATER_WINDOW,
               "window_days": int(early.window_days.iloc[0]), "model_features": MODEL_FEATURES,
               "selected_kmeans_k": km["selected_k"], "selected_gmm_k": gmm["selected_k"],
               "selected_gmm_covariance": gmm["covariance_type"],
               "low_membership_confidence_share_below_0_7": float(probabilities.max(axis=1).lt(.7).mean()),
               "mean_max_membership_probability": float(probabilities.max(axis=1).mean()),
               "temporal": temporal}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    early.join(early_labels).join(probabilities).to_csv(processed / "early_customer_segments.csv")
    later.join(later_labels).to_csv(processed / "later_customer_segments.csv")
    names = dict(zip(gmm["model"].predict(gmm["values"]), early_labels.to_numpy()))
    names = {int(key): value for key, value in names.items()}
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump({"scaler": gmm["scaler"], "model": gmm["model"], "segment_names": names,
                 "feature_names": MODEL_FEATURES, "window_days": int(early.window_days.iloc[0])},
                artifacts / "segmentation.joblib")
    return summary


def make_figures(root, purchases, early, gmm, labels, profiles, transitions, cohort, bootstrap):
    destination = Path(root) / "results/figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150})

    def save(fig, name):
        fig.savefig(destination / name, bbox_inches="tight")
        plt.close(fig)

    monthly = purchases.set_index("InvoiceDate").revenue.resample("MS").sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values / 1000, color=COLORS[0], marker="o", markersize=4)
    ax.set(title="Monthly UK merchandise purchases", ylabel="Gross spending (GBP thousands)", xlabel="Month")
    ax.text(.01, .98, "December 2011 has partial coverage", transform=ax.transAxes, va="top", fontsize=9)
    fig.autofmt_xdate()
    save(fig, "01_monthly_purchases.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for covariance, table in gmm["candidates"].groupby("covariance_type"):
        axes[0].plot(table.k, table.bic, marker="o", label=covariance)
        axes[1].plot(table.k, table.validation_log_likelihood, marker="o", label=covariance)
    axes[0].set(title="Mixture model selection", xlabel="Components", ylabel="Training BIC (lower is better)")
    axes[1].set(title="Independent customer holdout", xlabel="Components", ylabel="Mean log-likelihood (higher is better)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "02_gmm_selection.png")

    projection = PCA(n_components=2).fit(gmm["values"])
    points = projection.transform(gmm["values"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for number, name in enumerate(sorted(labels.unique())):
        mask = labels.eq(name).to_numpy()
        ax.scatter(points[mask, 0], points[mask, 1], s=12, alpha=.45, color=COLORS[number], label=name)
    ax.set(title="Customer segments: PCA visualization", xlabel=f"PC1 ({projection.explained_variance_ratio_[0]:.0%} variance)",
           ylabel=f"PC2 ({projection.explained_variance_ratio_[1]:.0%} variance)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Segment")
    save(fig, "03_customer_segments.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(profiles.index, profiles.customer_share * 100, color=COLORS[:len(profiles)])
    axes[0].set(title="Customer share", ylabel="Percent")
    axes[1].bar(profiles.index, profiles.median_spend_gbp, color=COLORS[:len(profiles)])
    axes[1].set(title="Median six-month spending", ylabel="GBP")
    errors = np.vstack([(profiles.repeat_purchase_rate_90d-profiles.repeat_rate_wilson_low)*100,
                        (profiles.repeat_rate_wilson_high-profiles.repeat_purchase_rate_90d)*100])
    axes[2].bar(profiles.index, profiles.repeat_purchase_rate_90d*100, color=COLORS[:len(profiles)], yerr=errors, capsize=3)
    axes[2].set(title="Purchased again within 90 days", ylabel="Percent", ylim=(0, 105))
    fig.tight_layout()
    save(fig, "04_segment_profiles.png")

    rows = sorted(labels.unique())
    columns = rows + ["No purchases in later window"]
    matrix = cohort.reindex(index=rows, columns=columns, fill_value=0)
    share = matrix.div(matrix.sum(axis=1), axis=0)*100
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(j, i, f"{share.iloc[i,j]:.1f}%\n(n={matrix.iloc[i,j]})", ha="center", va="center",
                    color="white" if share.iloc[i,j] > 55 else "#172554", fontsize=9)
    ax.set_xticks(range(len(columns)), rows+["No purchases\nin later window"])
    ax.set_yticks(range(len(rows)), rows)
    ax.set(title="Early customers: later segment or no observed purchases", xlabel="Later-window activity", ylabel="Early segment")
    fig.colorbar(im, ax=ax, label="Percent of early segment")
    save(fig, "05_customer_movement.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([bootstrap[x].ari for x in bootstrap], tick_labels=list(bootstrap), showmeans=True)
    axes[0].set(title="Bootstrap model robustness", ylabel="Adjusted Rand index", ylim=(-.05, 1.05))
    confidence = membership(gmm, early).max(axis=1)
    axes[1].hist(confidence, bins=np.linspace(0, 1, 21), color=COLORS[1], edgecolor="white")
    axes[1].axvline(.7, color=COLORS[4], linestyle="--", label="Illustrative 0.7 threshold")
    axes[1].set(title="Mixture membership confidence", xlabel="Maximum posterior membership", ylabel="Customers")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "06_robustness_and_confidence.png")
