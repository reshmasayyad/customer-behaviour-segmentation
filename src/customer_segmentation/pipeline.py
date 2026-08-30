"""One reproducible entry point for the complete analysis."""
from pathlib import Path
from .data import prepare_data
from .features import customer_features, future_outcomes, EARLY_WINDOW, LATER_WINDOW
from .models import fit_kmeans_candidates, fit_gmm_candidates
from .evaluation import segment_names, label_series, bootstrap_stability, segment_profiles, temporal_comparison, regularization_sensitivity
from .reporting import save_results, make_figures


def run_analysis(root, repetitions=20, verbose=True):
    root = Path(root)
    def progress(message):
        if verbose:
            print(message, flush=True)
    progress("Loading and cleaning the two retail worksheets...")
    purchases, audit = prepare_data(root)
    early = customer_features(purchases, *EARLY_WINDOW)
    later = customer_features(purchases, *LATER_WINDOW)
    progress(f"Early customers: {len(early):,}; later customers: {len(later):,}")
    km = fit_kmeans_candidates(early)
    progress(f"K-means selected k={km['selected_k']}")
    gmm = fit_gmm_candidates(early)
    progress(f"GMM selected k={gmm['selected_k']}, covariance={gmm['covariance_type']}")
    names = segment_names(gmm["model"].predict(gmm["values"]), early)
    early_labels = label_series(gmm, early, names)
    later_labels = label_series(gmm, later, names)
    profiles = segment_profiles(early, early_labels, future_outcomes(purchases, early.index, EARLY_WINDOW[1]))
    later_profiles = segment_profiles(later, later_labels, future_outcomes(purchases, later.index, LATER_WINDOW[1]))
    progress(f"Bootstrap robustness: {repetitions} refits per model...")
    bootstrap = {"KMeans": bootstrap_stability(km, early, repetitions),
                 "GMM": bootstrap_stability(gmm, early, repetitions)}
    transitions, cohort, temporal = temporal_comparison(early_labels, later_labels)
    summary = save_results(root, early, later, km, gmm, early_labels, later_labels, profiles,
                           later_profiles, bootstrap, transitions, cohort, temporal)
    sensitivity = regularization_sensitivity(gmm, early)
    sensitivity.to_csv(root / "results/regularization_sensitivity.csv", index=False)
    make_figures(root, purchases, early, gmm, early_labels, profiles, transitions, cohort, bootstrap)
    progress("Saved aggregate results, figures, customer-level local tables, and fitted model.")
    return {"purchases": purchases, "audit": audit, "early": early, "later": later, "km": km, "gmm": gmm,
            "early_labels": early_labels, "later_labels": later_labels, "profiles": profiles,
            "later_profiles": later_profiles, "bootstrap": bootstrap, "transitions": transitions,
            "cohort": cohort, "summary": summary, "sensitivity": sensitivity}
