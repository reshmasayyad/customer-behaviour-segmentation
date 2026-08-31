"""Generate an explanatory notebook using the tested analysis functions."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Customer Behaviour Segmentation and Stability Analysis

**Question:** Which customer purchasing patterns are useful to describe, how robust are the groups, and how do customers move between them over time?

We use real transactions from UCI Online Retail II. The analysis compares a K-means baseline with a probabilistic Gaussian mixture model (GMM), then separates **model robustness** from **changes in customer behaviour**. Subsequent purchases help interpret the groups without being used to build them.

This is an observational analytics project. It does not estimate the causal effect of marketing, claim labelled segment accuracy, or identify permanently churned customers.

**Run:** extract the repository, install `requirements.txt`, and run this notebook from `notebooks/` or the repository root. The first run downloads the official workbook and reads both worksheets; subsequent runs use a local cleaned-data cache. The data-loading step can take several minutes. Computation is CPU-only.
""")
code("""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / "src/customer_segmentation").exists():
    ROOT = ROOT.parent
assert (ROOT / "src/customer_segmentation").exists(), "Open inside the extracted repository."
sys.path.insert(0, str(ROOT / "src"))
from customer_segmentation.data import prepare_data
from customer_segmentation.features import customer_features, future_outcomes, MODEL_FEATURES, EARLY_WINDOW, LATER_WINDOW
from customer_segmentation.models import fit_kmeans_candidates, fit_gmm_candidates, membership, clustering_metrics
from customer_segmentation.evaluation import segment_names, label_series, bootstrap_stability, segment_profiles, temporal_comparison, regularization_sensitivity
from customer_segmentation.reporting import save_results, make_figures
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
""")
md("""
## 1. Load both transaction years and audit the cleaning

Source: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), Daqing Chen, DOI [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), CC BY 4.0.

The workbook contains 2009–2010 and 2010–2011 worksheets. We do not read just the first sheet. Sequential cleaning removes exact duplicates, invalid/missing customer IDs, cancellation invoices, non-positive quantity or price, and non-merchandise stock codes. We focus on UK customers to reduce geographic heterogeneity.

**Important definition:** monetary value is *gross positive merchandise spending*, not net revenue or profit. Returns are excluded rather than subtracted. Exact duplicates may sometimes be legitimate repeated invoice lines; removing them is a documented assumption. The numeric merchandise-code rule also excludes service and adjustment entries and is not a universal retail taxonomy.
""")
code("""
purchases, audit = prepare_data(ROOT)
display(audit)
manifest = json.loads((ROOT / "results/data_manifest.json").read_text())
display(pd.Series(manifest, name="Data provenance"))
print("Clean purchase columns:", purchases.columns.tolist())
""")
md(r"""
## 2. Construct RFM and product-diversity features

For customer $i$, using purchases only inside the observation window:

- $R_i$ = calendar days since the latest purchase, measured at the window end.
- $F_i$ = number of **distinct invoices**, not the number of product lines.
- $M_i = \sum_j q_{ij}p_{ij}$ = gross merchandise spending in GBP.
- $U_i$ = number of distinct merchandise product codes purchased.

Average order value $M_i/F_i$ is useful for profiles, but is not included as another modelling feature because it is derived from monetary value and frequency. This avoids unnecessarily duplicating the same information.

**Early window:** 1 December 2009 up to, but excluding, 1 June 2010. **Later window:** 1 December 2010 up to, but excluding, 1 June 2011. Both cover 182 days and similar seasons. This reduces, but does not eliminate, calendar effects.

Only customers with purchases in a window have a feature row for that window. Customers without later-window purchases are reported separately, rather than being silently removed or assigned artificial zero-spending active profiles.
""")
code("""
early = customer_features(purchases, *EARLY_WINDOW)
later = customer_features(purchases, *LATER_WINDOW)
display(pd.DataFrame({"period": ["Early", "Later"], "active_customers": [len(early), len(later)],
                      "window_days": [early.window_days.iloc[0], later.window_days.iloc[0]]}))
display(early[MODEL_FEATURES + ["average_order_gbp"]].head().reset_index(drop=True))
display(early[MODEL_FEATURES].describe().T)
""")
md(r"""
## 3. Compare a K-means baseline

Raw spending and order counts are skewed. Apply $\log(1+x)$ to each feature, then standardize it. K-means minimizes within-cluster squared Euclidean distance:

$$\min_{\mu_1,\ldots,\mu_K}\sum_i\min_k\|z_i-\mu_k\|^2.$$

We reserve 30% of early-window customers for model selection. Scaling is fitted only to the remaining 70%. Candidate $K$ values are 2 through 6. Select the highest holdout silhouette among candidates with every cluster represented and at least 1% of holdout customers in the smallest cluster. Then refit the selected configuration to all early-window customers.

This is a customer holdout for unsupervised selection, not a future-time test. The later year remains untouched until the temporal analysis. Silhouette measures geometric separation; it is not customer-segment accuracy.
""")
code("""
km = fit_kmeans_candidates(early)
display(km["candidates"])
print("Selected K-means cluster count:", km["selected_k"])
""")
md(r"""
## 4. Fit a probabilistic Gaussian mixture

A Gaussian mixture models the transformed customer features as:

$$p(z_i)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(z_i\mid\mu_k,\Sigma_k).$$

The model gives posterior component membership:

$$\gamma_{ik}=\frac{\pi_k\mathcal{N}(z_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(z_i\mid\mu_j,\Sigma_j)}.$$

We compare 2–6 components with diagonal and full covariance. Training BIC is $-2\log L+p\log n$: lower values favour fit after penalizing complexity. Holdout log-likelihood is an additional diagnostic. All BIC comparisons use the same customer sample and feature transformation. BIC should not be compared across different feature dimensions.

We require convergence and minimum holdout cluster sizes. A covariance floor of 0.01 in standardized feature units reduces component collapse around discrete order counts. This is a modelling assumption that will be stress-tested. GMMs approximate a dataset containing discrete counts; they do not reveal objectively true or automatically calibrated business groups.

The selected component count is the best **within this bounded search**, not a proof of the globally optimal number of segments. Multiple initializations reduce, but do not eliminate, local-optimum risk.
""")
code("""
gmm = fit_gmm_candidates(early)
display(gmm["candidates"].sort_values("bic"))
print("Selected mixture:", gmm["selected_k"], "components;", gmm["covariance_type"], "covariance")
names = segment_names(gmm["model"].predict(gmm["values"]), early)
early_labels = label_series(gmm, early, names)
later_labels = label_series(gmm, later, names)
probabilities = membership(gmm, early)
assert np.allclose(probabilities.sum(axis=1), 1)
display(probabilities.head().reset_index(drop=True))
print("Share below illustrative 0.7 maximum membership:", f"{probabilities.max(axis=1).lt(.7).mean():.2%}")
""")
md("""
## 5. Profile the segments and examine subsequent purchasing

Display labels S1, S2, … are ordered by median early-window spending. They are convenient IDs, not a claim that every dimension increases with segment number. A group with high spending but low product diversity may have different needs from a frequent, diverse-basket group.

For each customer, we observe purchases in the following 90 days. Customers with no follow-up purchase receive zero future orders and spending. These outcomes are joined only after clustering and are not used for model selection. Repeat-purchase rates are descriptive; they do not prove an intervention would work. Wilson intervals summarize binomial sampling uncertainty under independent-customer assumptions.
""")
code("""
early_outcomes = future_outcomes(purchases, early.index, EARLY_WINDOW[1])
later_outcomes = future_outcomes(purchases, later.index, LATER_WINDOW[1])
profiles = segment_profiles(early, early_labels, early_outcomes)
later_profiles = segment_profiles(later, later_labels, later_outcomes)
display(profiles[["customers", "median_recency_days", "median_orders", "median_spend_gbp",
                  "median_products", "customer_share", "spend_share", "repeat_purchase_rate_90d"]])
display(later_profiles[["customers", "median_spend_gbp", "repeat_purchase_rate_90d"]])
""")
md("""
## 6. Measure model robustness using bootstrap refits

Resample early-window customers with replacement 20 times. Refit the feature scaler and the chosen model configuration each time. Compare assignments on the same fixed anchor customers using adjusted Rand index (ARI), which is unaffected by permutations of numeric component labels.

ARI near 1 indicates nearly identical partitions; near 0 indicates agreement comparable to chance under the metric's assumptions. Selected K and covariance type remain fixed. Therefore this tests robustness of a configuration, not uncertainty about the entire model-selection procedure. The 5th–95th percentile range is an empirical robustness range, **not a population confidence interval**.
""")
code("""
bootstrap = {"KMeans": bootstrap_stability(km, early, repetitions=20),
             "GMM": bootstrap_stability(gmm, early, repetitions=20)}
stability = pd.DataFrame([
    {"model": model, "median_ARI": values.ari.median(), "5th_percentile": values.ari.quantile(.05),
     "95th_percentile": values.ari.quantile(.95), "all_converged": values.converged.all()}
    for model, values in bootstrap.items()
])
display(stability)
""")
md("""
## 7. Measure customer movement using a frozen model

Apply the original early-window scaler and model to later-window features. We do **not** refit the later year or rename its component IDs independently. This keeps the meaning of segment IDs comparable.

Two views matter: transitions among customers active in both windows, and activity of the entire early cohort including those with no later-window purchases. Later active customers missing from the early window are labelled *not observed in the early window*, rather than assumed to be first-time customers. No later-window purchase does not establish permanent churn.

Movement may reflect genuine behaviour, remaining calendar differences, changes in the retailer, sample coverage, or modelling limitations. It should not be attributed to one cause without further evidence.
""")
code("""
transitions, cohort, temporal = temporal_comparison(early_labels, later_labels)
display(pd.Series(temporal, name="Temporal summary"))
display(transitions)
display(cohort)
assert cohort.to_numpy().sum() == len(early)
""")
md("""
## 8. Stress-test covariance regularization

Order frequency and product diversity are counts. Nearly collapsed Gaussian components can produce overly confident memberships. Refit the same selected K and covariance family with several covariance floors, then compare assignments with the primary model.

A change in ARI or membership confidence demonstrates sensitivity. High posterior membership alone is not evidence of reliable real-world classification. There are no labelled segments against which calibration can be validated.
""")
code("""
sensitivity = regularization_sensitivity(gmm, early)
display(sensitivity)
""")
md("""
## 9. Save reproducible outputs and inspect the figures

Aggregate tables and figures go into `results/`. Local customer-level assignments go into `data/processed/`, which is excluded from Git. The fitted scaler and mixture are saved together with their feature definitions and display-label mapping in `artifacts/segmentation.joblib`.

PCA is used only to visualize the four-dimensional features; it is not the clustering input, and a 2D plot cannot prove separation in the original space.
""")
code("""
summary = save_results(ROOT, early, later, km, gmm, early_labels, later_labels, profiles,
                       later_profiles, bootstrap, transitions, cohort, temporal)
sensitivity.to_csv(ROOT / "results/regularization_sensitivity.csv", index=False)
make_figures(ROOT, purchases, early, gmm, early_labels, profiles, transitions, cohort, bootstrap)
display(pd.read_csv(ROOT / "results/final_model_comparison.csv"))
for filename in ["01_monthly_purchases.png", "02_gmm_selection.png", "03_customer_segments.png",
                 "04_segment_profiles.png", "05_customer_movement.png", "06_robustness_and_confidence.png"]:
    display(Image(filename=str(ROOT / "results/figures" / filename)))
""")
md("""
## 10. Demonstrate scoring another customer-feature table

The example rows below are **synthetic demonstrations**, not additional observed customers. The saved model can score a feature table without retraining. Inputs must follow the same cleaning, feature definitions, units, and 182-day observation window. A different retailer or materially different distribution requires fresh validation.

From the repository root:

```bash
python scripts/score_customers.py data/example_customer_features.csv results/example_scores.csv
```
""")
code("""
sys.path.insert(0, str(ROOT / "scripts"))
from score_customers import score
display(score(ROOT / "data/example_customer_features.csv", ROOT / "results/example_scores.csv",
              ROOT / "artifacts/segmentation.joblib"))
""")
md("""
## 11. Findings and business interpretation

The following statements are generated from the measured results, so they update if the data or model changes. The mix of geometric separation, statistical fit, and robustness helps decide whether broad or detailed groups are suitable for a particular use case. GMM is not presumed to outperform K-means on every measure.
""")
code("""
largest_spend_segment = profiles.spend_share.idxmax()
row = profiles.loc[largest_spend_segment]
print(f"The bounded selection chose {km['selected_k']} K-means groups and {gmm['selected_k']} GMM components.")
print(f"{largest_spend_segment}: {row.customer_share:.1%} of early customers account for {row.spend_share:.1%} of observed gross spending.")
print(f"Observed 90-day repeat-purchase rate in that segment: {row.repeat_purchase_rate_90d:.1%}.")
print(f"Among {temporal['active_in_both']:,} customers active in both windows, {temporal['same_segment_share_among_active_in_both']:.1%} kept the same frozen-model segment.")
print(f"{temporal['no_purchase_in_later_window']:,} early customers had no purchases in the later window; this is not a permanent-churn label.")
display(stability)
""")
md("""
**Possible actions to investigate:** offer onboarding support to one-order groups, study retention for infrequent groups, and investigate whether concentrated high-spending customers need wholesale support. These are hypotheses requiring controlled evaluation; this analysis does not measure campaign uplift or realized revenue gains.

**Limitations:** one retailer; gross purchases excluding returns; discrete features approximated by Gaussians; repeated-line cleaning assumptions; limited model search; high-value customers retained rather than automatically treated as errors; temporal comparisons restricted by customer coverage; and no external segment ground truth. Seasonal matching does not make the two years identical.

**Useful extensions:** compare count-aware mixture models, measure sensitivity to returns and duplicate handling, repeat the analysis over more rolling windows, examine cold-start customers, and evaluate any proposed marketing intervention with a properly designed experiment.

**References:**
- [UCI dataset and license](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- [Scikit-learn Gaussian mixture selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
- [Clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}}
path = ROOT / "notebooks/customer_segmentation.ipynb"
nbf.write(notebook, path)
print(f"Created {path} with {len(cells)} cells")
