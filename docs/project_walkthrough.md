# Understanding the project

## The question

An online retailer serves customers with different purchasing patterns. We want to describe those patterns, evaluate how dependable the grouping is, and see whether customers retain the same pattern a year later. We then examine subsequent purchasing to assess whether the groups carry descriptive business information.

## What one data row means

A transaction row is one product line on an invoice. An invoice can contain many rows. A customer can have many invoices. Therefore frequency must count distinct invoices, while spending sums quantity multiplied by unit price. Counting transaction rows as orders would inflate frequency for customers with diverse baskets.

## Why the feature window matters

We fix a six-month period and use purchases only inside it. Recency is measured at the exclusive end of that period. A purchase exactly at the cutoff belongs to the follow-up, not the features. Using future purchases to build the current profile would leak information.

Recency, frequency, monetary value and product diversity describe how recently, often, extensively and broadly a customer buys. Average basket value is reported but not separately included as a fifth model feature, because it is computed from spending and frequency.

## Why transform and scale

Money and order counts are heavily skewed. The `log1p` transformation compresses extremes while preserving order. Standardization prevents pounds from dominating days and counts merely because their numerical scale is larger. Neither step makes the data exactly Gaussian.

## What K-means and a GMM do differently

K-means partitions customers using distances to centroids. Its two selected groups give a broad description. The Gaussian mixture fits several component distributions; its five selected components give a finer description and posterior membership values. Both use the same four transformed features.

The finer mixture does not automatically have better geometric separation or robustness. Its posterior can be highly confident under an imperfect distributional model. Frequency is discrete, and Gaussian covariance collapse near specific order counts is a concrete risk. We regularize covariance and test sensitivity rather than presenting membership values as ground truth.

## How model selection works

Only the early-window customer population is used. A random 70/30 customer split separates fitting from selection diagnostics. The K-means configuration is selected using holdout silhouette. GMM BIC balances training likelihood against parameter count, with holdout likelihood reported independently. Convergence and minimum component-size checks screen candidates.

The chosen configurations are subsequently refitted on all early customers. The later year is reserved for temporal analysis. This customer holdout and later-year application answer different questions; neither is a supervised accuracy benchmark.

## What the stability result means

Bootstrap resampling asks: if we had observed a slightly different early customer sample, would the selected configuration assign the same anchor customers similarly? The scaler and model are refitted each time. ARI handles arbitrary numeric component labels without requiring a label-matching algorithm.

K and the covariance form stay fixed during these checks. The results therefore assess a particular configuration. They do not claim that the selected number of components would be unchanged under every possible dataset.

## What customer movement means

The original scaler and model score later-window features. A customer moving from S1 to S4 has a changed profile according to a fixed definition. This is different from independently clustering both years, where component IDs could have unrelated meanings.

The analysis includes an explicit no-purchases destination for early customers with no later activity. Restricting all reporting to customers present in both periods would hide these people and create an overly favourable picture of stability. Later-only customers are described as not observed in the early window; we cannot assume they are new to the retailer.

## Why observe another 90 days

Subsequent purchases help interpret the groups. For instance, one-order groups may buy again less often than frequent groups. These outcomes do not enter feature construction or model selection. The differences remain observational and do not show that an email campaign would increase purchases.

## How to discuss the results in an interview

Be ready to explain why K-means was more stable, why a GMM was still useful, and why the same-segment percentage is computed among customers active in both periods while the full cohort is reported separately. Read `results/segment_profiles.csv`, `results/stability_summary.csv`, and `results/regularization_sensitivity.csv` before making numerical claims.

Use ARI as a robustness metric, not accuracy. Use observed repeat purchasing as descriptive evidence, not causal uplift. Describe the project as customer segmentation and temporal analysis, not a deployed churn prediction or recommendation system.

## What would make the next version better

Test duplicate and return-handling assumptions, compare a model designed for mixed counts and continuous features, repeat over more rolling windows, and investigate high-spending concentrated-basket accounts separately. Marketing recommendations should be evaluated with a controlled intervention, not inferred from cluster averages.
