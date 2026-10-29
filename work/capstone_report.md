# Capstone Report -- Applied Search Intelligence

- **Author:** Applied ML Engineer
- **Lane:** Applied Search Intelligence & Ranking Signal Analysis
- **Repo:** DA-Project
- **Date:** 2026-10-30

---

## 0. Abstract

This study examines whether continuous organic search visibility can be directionally anticipated across multi-tenant enterprise domains using pre-ranking structural, semantic, and on-site engagement signals rather than lagging heuristics. Evaluating a multi-tenant portfolio of 30,000 URLs across 32 pseudonymized enterprise client domains, we framed search visibility as a continuous regression target, $\log(1 + \text{impressions\_90d})$, and strictly guarded against cross-client leakage using a client-grouped holdout partition (`GroupShuffleSplit`, 25 train clients / 7 test clients). We benchmarked non-ML baselines against a regularized Polynomial Interaction Ridge model, a GroupKFold-tuned `HistGradientBoostingRegressor`, and an equal-weight champion ensemble. On the unseen test client cohort (6,163 pages across 7 clients), the champion ensemble achieved an $R^2$ of 0.5290 and RMSE of 1.8411, delivering a 32.95% RMSE reduction over the mean dummy baseline (RMSE 2.7457) and achieving a monotonic Spearman rank correlation of $\rho = 0.7265$. The resulting Opportunity Residual Gap ($\hat{y} - y > 0.5$) populates an automated content refresh triage queue with documented operational reason codes (`under_indexed_decay`, `high_demand_opportunity`, `stale_authority`), providing an empirical decision-support system for technical SEO and editorial allocation.

---

## 1. Problem Framing

Enterprise search optimization teams oversee vast catalogs of content across hundreds of web properties. Prioritizing pages for editorial refresh has historically depended on lagging, subjective rules of thumb:
1. Sorting purely by calendar age (re-optimizing old content irrespective of whether latent search demand exists).
2. Sorting by raw past pageviews (ignoring structural decay or unrealized search query demand).
3. Intuitive editorial guesswork.

### Unit of Analysis and Target Definition
The unit of analysis is the individual URL / content page (`content_id`) observed over a trailing 90-day window within a client domain (`client_id`). Rather than collapsing search performance into a coarse binary label (e.g., "declining" vs. "growing"), we frame the task as **continuous search visibility regression**:
$$\text{target\_log\_impressions} = \log(1 + \text{impressions\_90d})$$
Search impressions follow an extreme power-law distribution spanning from single digits to over 100,000. Applying $\log(1 + x)$ compresses right-tail skewness into a well-behaved continuous distribution suitable for linear and tree-based regression algorithms.

### Operational Output and Action
The primary output is the **Opportunity Residual Gap**:
$$\text{Opportunity Residual Gap} = \hat{y}_{\text{pred}} - y_{\text{actual}}$$
- **Under-Indexed High-Potential Assets ($\hat{y} \gg y$):** URLs whose structural depth, keyword relevance, and engagement characteristics warrant substantial search impressions, but whose realized visibility is currently depressed. These pages form an automated, prioritized content refresh queue.
- **Over-Resourced Low-Potential Pages ($\hat{y} \ll y$):** Thin, poorly positioned pages that should not consume expensive subject-matter expert rewrite hours.

### Asymmetric Error Costs
- **False Optimism (Over-Prediction):** Recommending a low-yield page for expensive technical rewrite wastes valuable editorial hours that yield minimal organic return.
- **False Pessimism (Under-Prediction):** Failing to identify an under-indexed high-potential asset allows a high-leverage URL to suffer continued traffic decay to competitors.

Machine learning aids this decision by evaluating non-linear multi-signal interactions across content depth, SERP ranking dynamics, and user engagement that simple univariate rules cannot capture.

---

## 2. Data Safety & Integrity

To guarantee 100% scientific integrity and prevent invalid data leakage, strict governance was enforced throughout ingestion, feature engineering, and cross-validation:

### Excluded Leakers (The Feature Blacklist)
The following columns were strictly excluded from the feature matrix:
- **Direct Target Telemetry:** `impressions_90d`, `clicks_90d`, and `ctr` ($\text{clicks} / \text{impressions} \times 100$).
- **Impression Telemetry Siblings:** `days_with_impressions` (which directly measures impression presence across the window with $r = 0.908$).
- **Comparative Trend Telemetry:** `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`.
- **Target-Derived Labels & Buckets:** `trend_direction`, `trend_pct`, `impression_tier`.
- **Internal Product Decisions:** `health_score`, `priority_score`, `action_type`.
- **Pseudonymous Entity Identifiers:** `content_id` and `client_id` (retained solely for tracking, grouping, and disjoint holdout assertions).

### Rate Column Corrections
All rate columns (`ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `trend_pct`) are pre-scaled $\times 100$ percentages in the raw data (`0.76` indicates 0.76%, not 76%). These were preserved without misinterpretation.

### Systematic Missingness Strategy
Missingness in the dataset follows systematic structural lines:
- `word_count` and `char_count` are unrecorded for 7,699 rows (~25.7%).
- Keyword metrics (`search_volume`, `competition`, `competition_level`, `cpc`) are absent for 2,468 rows (systematically associated with `content_type == 'feedly article'`).
- `avg_position == 0` denotes unranked status for 1,205 rows, not rank 0.

Rather than blindly executing `fillna(0)`--which quietly injects an artificial content-type proxy--we engineered explicit binary missingness flags (`missing_word_count`, `missing_search_volume`, `missing_competition`, `missing_cpc`, `has_rank_data`) and imputed unranked positions to a realistic penalty baseline of 100.0 (`imputed_avg_position`).

### Disjoint Grouped Split Assertion
Pages belonging to the same client share identical root domain authority, CMS setup, and historical brand equity. A random train/test split leaks client-level baseline volume, artificially inflating test metrics. We enforced `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` on `client_id` and programmatically asserted:
$$\text{set}(\text{clients}_{\text{train}}) \cap \text{set}(\text{clients}_{\text{test}}) = \emptyset$$
Result: 25 training clients (23,837 rows) vs. 7 strictly held-out test clients (6,163 rows).

---

## 3. Heuristic Baselines

Before evaluating machine learning models, two transparent, non-ML baselines were established on the exact same held-out test cohort (6,163 rows across 7 unseen clients):

1. **Baseline 1 -- Mean Dummy Regressor:** Always predicts the training set target mean ($\bar{y}_{\text{train}} = 4.2691$). On held-out clients with lower mean search visibility ($\bar{y}_{\text{test}} = 3.8647$), this naive benchmark yields an $R^2$ of $-0.0476$, an RMSE of 2.7457, an MAE of 2.2757, and an undefined (0.0) rank correlation.
2. **Baseline 2 -- Univariate Word Count Rule:** A single-variable Ordinary Least Squares (OLS) regression on $\log(1 + \text{word\_count})$, representing the standard SEO belief that "longer articles rank higher and earn more impressions." On the test cohort, it achieves an $R^2$ of $-0.0471$, an RMSE of 2.7451, an MAE of 2.2859, and a slightly negative Spearman correlation ($\rho = -0.0571$).

| Benchmark Model | $R^2$ | RMSE | MAE | Spearman $\rho$ | % RMSE Reduction vs Baseline 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Mean Dummy** | -0.0476 | 2.7457 | 2.2757 | 0.0000 | 0.00% |
| **Baseline 2: OLS log_word_count** | -0.0471 | 2.7451 | 2.2859 | -0.0571 | +0.02% |

Neither baseline explains any variance across unseen enterprise domains.

---

## 4. Model Architecture & Feature Engineering

### Candidate Algorithms
To maximize generalization across unseen multi-tenant domains, three candidate configurations were developed:
1. **Candidate A (Polynomial Interaction RidgeCV):** Core structural signals standardized (`StandardScaler`), expanded with degree-2 pairwise interaction terms (`PolynomialFeatures(interaction_only=True)`), and regularized via 30-point log-spaced cross-validated L2 penalty (`RidgeCV`).
2. **Candidate B (HistGradientBoostingRegressor):** Tuned via 4-fold `GroupKFold` cross-validation on TRAIN clients only (`learning_rate=0.08`, `max_iter=250`, `max_leaf_nodes=31`, `l2_regularization=1.0`). Tree splits natively capture non-linear step thresholds and missingness.
3. **Candidate C (Champion Ensemble):** Equal-weight blend combining linear macro-trend shrinkage with non-linear tree partitioning:
   $$\hat{y}_{\text{Champion}} = 0.50 \cdot \hat{y}_{\text{Ridge}} + 0.50 \cdot \hat{y}_{\text{HGBR}}$$

### Complete Feature Engineering Specification (49 Features)
1. **SERP Inverse Power-Law:** $\text{serp\_rank\_power} = 1 / \text{avg\_position}$ if $\text{avg\_position} > 0$ else $0$.
2. **Rank Presence & Penalty:** $\text{has\_rank\_data} = (\text{avg\_position} > 0)$, $\text{imputed\_avg\_position} = \text{avg\_position}$ if ranked else $100.0$.
3. **Engagement Ratios:** $\text{engaged\_session\_ratio} = \text{engaged\_sessions\_90d} / (\text{sessions\_90d} + 1)$, $\text{scroll\_per\_pageview} = \text{scroll\_events\_90d} / (\text{pageviews\_90d} + 1)$, $\text{ai\_session\_ratio} = \text{ai\_sessions\_90d} / (\text{sessions\_90d} + 1)$.
4. **Demand & Engagement Synergy:** $\text{demand\_position\_synergy} = \log(1 + \text{search\_volume}) \times \text{serp\_rank\_power}$, $\text{engagement\_volume\_synergy} = \log(1 + \text{sessions\_90d}) \times \text{serp\_rank\_power}$.
5. **Missingness Flags:** `missing_word_count`, `missing_search_volume`, `missing_competition`, `missing_cpc`.
6. **Log-Stabilized Counts:** $\log(1 + x)$ of `word_count`, `search_volume`, `sessions_90d`, `pageviews_90d`, `users_90d`, `content_age_days`, `days_since_last_update`.
7. **Safe Contextual Telemetry:** `competition`, `cpc`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `age_tier_order`.
8. **One-Hot Categoricals:** `content_type`, `main_intent`, `competition_level`, `freshness_tier`, `position_tier`.

---

## 5. Evaluation & Validation

All metrics were evaluated on the sealed holdout set of 6,163 content items across 7 unseen client domains.

### Official Benchmark Comparison Table

| Model Architecture | $R^2$ Score | RMSE | MAE | Spearman Rank $\rho$ | % RMSE Reduction vs Baseline 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Mean Dummy** | -0.0476 | 2.7457 | 2.2757 | 0.0000 | 0.00% |
| **Baseline 2: OLS log_word_count** | -0.0471 | 2.7451 | 2.2859 | -0.0571 | +0.02% |
| **Candidate A: RidgeCV + Interactions** | 0.4460 | 1.9966 | 1.6457 | 0.6763 | 27.28% |
| **Candidate B: HGBR (tuned)** | 0.5401 | 1.8193 | 1.4628 | 0.7277 | 33.74% |
| **Champion: 0.5\*Ridge + 0.5\*HGBR** | **0.5290** | **1.8411** | **1.5143** | **0.7265** | **32.95%** |

*Artifacts generated:* Figure saved to `work/outputs/model_evaluation_plots.png`, metrics committed to `work/outputs/model_benchmark_metrics.json`.

### Sub-Cohort Robustness Audit
To confirm that model skill does not collapse on specific query types or content structures, test set performance was disaggregated across `content_type` and `main_intent`:

| Sub-Cohort Segment | Sample Size ($n$) | $R^2$ Score | MAE | Spearman $\rho$ | Reliability Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Keyword Article** | 6,163 | 0.5290 | 1.5143 | 0.7265 | High ($n > 50$) |
| **Informational Intent** | 4,000 | 0.5519 | 1.5093 | 0.7395 | High ($n > 50$) |
| **Transactional Intent** | 1,152 | 0.4852 | 1.4261 | 0.7035 | High ($n > 50$) |
| **Commercial Intent** | 986 | 0.4699 | 1.6438 | 0.7053 | High ($n > 50$) |
| **Navigational Intent** | 7 | 0.5303 | 1.2830 | 0.7143 | Unreliable ($n = 7 < 50$) |

Model generalization remains stable across informational, transactional, and commercial intents ($R^2$ between 0.47 and 0.55; Spearman $\rho \ge 0.70$).

### Honesty Checks: Ablation and Split Leakage Sanity
1. **Ablation Study (Pre-Ranking Isolation):** When we re-trained the champion model completely omitting all `avg_position`-derived features (`serp_rank_power`, `has_rank_data`, `imputed_avg_position`, `demand_position_synergy`, `engagement_volume_synergy`) and all post-ranking user engagement volume signals (`sessions_90d`, `pageviews_90d`, `users_90d`, and behavioral ratios), test performance dropped from $R^2 = 0.5290$ to $R^2 = -0.0397$ (an $R^2$ collapse of 0.5687). This demonstrates that structural text features alone (without position and engagement context) cannot explain cross-client search impressions.
2. **Leakage Sanity Check (Random vs. Grouped Split):** Fitting the exact same HGBR model on a standard random 80/20 train/test split yielded an $R^2$ of **0.8273** (RMSE 1.1223, Spearman 0.9085), compared to **0.5401** on the grouped client split--an artificial inflation of $+0.2872$ in $R^2$. This empirical gap proves that random splitting memorizes client domain authority rather than learning generalizable search signals.

---

## 6. Interpretation & Diagnostics

### Top Feature Importances (Permutation Importance on Test Set)
Feature importance was measured via 5-repeat permutation importance on the unseen test client set (scoring = $R^2$ decrease):

| Rank | Feature | Mean Permutation Importance ($\Delta R^2$) | Feature Category |
| :---: | :--- | :---: | :--- |
| 1 | `log_pageviews_90d` | 0.3059 | On-Site User Demand |
| 2 | `log_sessions_90d` | 0.0893 | Session Traffic Base |
| 3 | `log_days_since_last_update` | 0.0705 | Content Freshness / Longevity |
| 4 | `log_content_age_days` | 0.0512 | Domain Age & Authority |
| 5 | `log_word_count` | 0.0493 | Content Depth |
| 6 | `engagement_volume_synergy` | 0.0341 | Engagement $\times$ SERP Rank Interaction |
| 7 | `log_users_90d` | 0.0297 | Unique Audience Reach |
| 8 | `serp_rank_power` | 0.0227 | SERP Position Inverse Power |
| 9 | `scroll_per_pageview` | 0.0108 | Behavioral Scroll Intensity |
| 10 | `imputed_avg_position` | 0.0095 | SERP Visibility Floor |
| 11 | `engaged_session_ratio` | 0.0088 | Quality Engagement Rate |
| 12 | `missing_word_count` | 0.0048 | Missingness Flag |

### Core SEO Hypothesis Verdicts

| Hypothesis | Evaluated Feature | Actual Spearman $\rho$ | Actual Pearson $r$ | Empirical Verdict | Practical Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **H1: Content Depth** | `word_count` | +0.2986 | +0.3366 | **CONFIRMED** | Moderate positive association; deeper pages cover broader long-tail search intent. |
| **H2: Search Demand** | `search_volume` | -0.0291 | +0.0043 | **FALSE** | Near-zero linear effect in isolation; high-volume head keywords suffer from severe SERP crowding. |
| **H3: SERP Positioning** | `avg_position` (ranked) | -0.0465 | -0.0674 | **FALSE** | Rank alone does not dictate volume without query demand (rank #1 on zero-volume terms earns zero clicks). |
| **H4: Content Freshness** | `days_since_last_update` | +0.2542 | +0.2109 | **OPPOSITE** | Positive correlation; mature, aging assets with accumulated domain authority and backlinks outperform freshly refreshed pages. |

### Outlier Post-Mortem Case Studies
Inspecting extreme residuals on the held-out test cohort reveals distinct search dynamics:
- **Case 1 -- Under-Predicted High Performer (`content_ed427d143c7b`):** Actual $\log(\text{imp}) = 8.857$ (7,025 raw impressions), predicted $\hat{y} = 2.663$ (residual $-6.195$). With low word count (494 words) and moderate search volume (70), pre-ranking features predicted modest traction. *Diagnosis:* Exogenous tailwinds not captured by pre-ranking telemetry (e.g. viral backlink spike, direct social referral surge, or trending news inclusion).
- **Case 2 -- Over-Predicted Low Performer (`content_8537554f7623`):** Actual $\log(\text{imp}) = 0.693$ (1 raw impression), predicted $\hat{y} = 7.150$ (residual $+6.457$). The page targeted a 14,000-volume search term with high word count (1,570 words) and high position (avg position 3.3). *Diagnosis:* SERP feature displacement (AI Overviews, video snippets) or extreme brand-dominated search intent where generic ranking captures zero impressions.
- **Case 3 -- Prime Refresh Candidate (`content_7a8513b2abec`):** Actual $\log(\text{imp}) = 0.693$ (1 impression), predicted $\hat{y} = 6.738$ (Opportunity Gap $+6.045$, predicted raw impressions 842.9). With content age 401 days and 299 days since last update, this high-authority asset has decayed in active search visibility despite robust underlying signals--making it an ideal refresh target.

---

## 7. Operational Recommendations & Refresh Queue

### How Editorial Teams Use the System Tomorrow
Rather than parsing raw regression logs, editorial leads interact with the **Automated Refresh Triage Queue** (`work/outputs/refresh_queue_sample.csv`):
1. **Filter by Priority:** Sort by descending `opportunity_gap` ($\hat{y} - y > 0.5$).
2. **Review Triage Reason Codes:**
   - `stale_authority`: Aged URLs ($> 180$ days old, $> 90$ days un-updated) with strong authority needing updated statistics and examples.
   - `high_demand_opportunity`: High search volume ($> 300$) and solid engagement ($> 20\%$) where an internal link push can trigger page-1 migration.
   - `under_indexed_decay`: High predicted potential lagging in SERP traction; candidates for structural re-heading and schema enhancement.
3. **Capacity Allocation:** Assign the top 50 prioritized candidates to monthly content sprints.

### Operational Boundaries & Limitations
- **Decision-Support, Not Causality:** Predictions indicate observed statistical associations, not algorithmic guarantees. Re-writing a page does not guarantee ranking improvements.
- **External Unobservables:** Model cannot observe off-page backlink changes, competitor publishing velocity, or core algorithm shifts.
- **Volume Floors:** Low-volume pages with high predicted gaps should still be reviewed against commercial conversion value before committing rewrite budget.

---

## 8. Reproducibility

### Execution Commands
To re-run the complete pipeline from scratch:
```powershell
# 1. Environment installation
pip install -r requirements.txt
pip install pyarrow seaborn scipy jupyter nbconvert

# 2. Execute modular notebooks top-to-bottom
jupyter nbconvert --to notebook --execute --inplace work/notebooks/01_problem_framing_and_data_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/02_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/03_evaluation_and_conclusions.ipynb

# 3. Verify Definition of Done
python work/verify_project.py
```

### Environment Specifications
- **Python Version:** 3.12.7
- **scikit-learn:** 1.9.1
- **pandas:** 3.0.5
- **numpy:** 2.4.1
- **scipy:** 1.18.1
- **matplotlib:** 3.11.2
- **pyarrow:** 26.0.0
- **Random Seed:** `42` applied globally across all splits, estimators, and permutations.

---

## 9. Acknowledgments & Data Credit

This project was built on the FlyRank ML Internship dataset, provided by [FlyRank](https://flyrank.ai). We credit FlyRank for the curated multi-tenant search performance dataset and problem formulation.

