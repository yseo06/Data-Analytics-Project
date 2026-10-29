# FlyRank Applied Search Intelligence: End-to-End Project Execution Summary

The **FlyRank Applied Search Intelligence** project has been executed across all phases (Phase 0 through Phase 4) in strict accordance with repository guidelines, data safety rules, and anti-leakage constraints. All 9 required deliverables exist, all 3 modular notebooks execute from top to bottom on a fresh kernel with zero errors, and [`work/verify_project.py`](file:///c:/Users/ACER/Documents/DA-Project/work/verify_project.py) confirms that **27 out of 27 checks pass**.

---

## 1. Official Benchmark Performance Table

All models were fitted strictly on the **25 training client domains (23,837 rows)** and evaluated on the **7 held-out client domains (6,163 rows)** with zero domain leakage:

| Model Architecture | $R^2$ Score | RMSE | MAE | Spearman Rank $\rho$ | % RMSE Reduction vs. Baseline 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Mean Dummy Regressor** | -0.0476 | 2.7457 | 2.2757 | 0.0000 | 0.00% |
| **Baseline 2: Univariate OLS (`log_word_count`)** | -0.0471 | 2.7451 | 2.2859 | -0.0571 | +0.02% |
| **Candidate A: Polynomial Interaction RidgeCV** | 0.4460 | 1.9966 | 1.6457 | 0.6763 | 27.28% |
| **Candidate B: Tuned `HistGradientBoostingRegressor`** | 0.5401 | 1.8193 | 1.4628 | 0.7277 | 33.74% |
| **Champion Ensemble: 0.5\*Ridge + 0.5\*HGBR** | **0.5290** | **1.8411** | **1.5143** | **0.7265** | **32.95%** |

**Key Finding:** The champion model reduces prediction error by **32.95%** over the naive mean baseline and achieves strong monotonic ranking discrimination ($\rho = 0.7265$). In contrast, univariate content depth (`log_word_count`) fails to explain any variance across unseen enterprise client domains ($R^2 = -0.0471$).

---

## 2. Statistical SEO Hypothesis Verdicts (Measured on Actual Data)

Correlations against the continuous search visibility target ($\log(1 + \text{impressions\_90d})$) yielded honest empirical verdicts based on actual measured values:

| Hypothesis | Feature Evaluated | Sample Size ($n$) | Spearman $\rho$ | Pearson $r$ | Empirical Verdict | Practical SEO Insight |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **H1: Content Depth** | `word_count` | 22,301 | **+0.2986** | +0.3366 | **CONFIRMED** | Moderate positive correlation ($p < 10^{-300}$). Longer, comprehensive articles systematically capture more long-tail search impressions. |
| **H2: Search Demand** | `search_volume` | 27,532 | **-0.0291** | +0.0043 | **FALSE** | Near-zero linear effect in isolation. High-volume head keywords suffer from severe SERP competition; volume alone does not translate to impressions without rank. |
| **H3: SERP Positioning** | `avg_position` (ranked rows $>0$) | 28,795 | **-0.0465** | -0.0674 | **FALSE** | Rank alone is uninformative in isolation: ranking #1 on a 10-volume query yields far fewer impressions than page 2 on a 50,000-volume query. |
| **H4: Content Freshness** | `days_since_last_update` | 30,000 | **+0.2542** | +0.2109 | **OPPOSITE** | Positive association ($p < 10^{-298}$). Mature URLs with accumulated domain authority, backlinks, and indexation longevity outperform freshly updated or newly published pages. |

---

## 3. Honesty & Ablation Checks

### Ablation Study (Pre-Ranking Isolation)
When the champion model was re-evaluated **without** `avg_position`-derived features (`serp_rank_power`, `has_rank_data`, `imputed_avg_position`, `demand_position_synergy`, `engagement_volume_synergy`) and **without** post-ranking engagement telemetry (`sessions_90d`, `pageviews_90d`, `users_90d`, and behavioral ratios):
- **Full Champion $R^2$:** `0.5290`
- **Ablation Champion $R^2$:** `-0.0397`
- **$R^2$ Drop:** **0.5687**

**Explanation:** Content signals alone (word count, categories, raw keyword stats) cannot predict cross-client impressions without knowing whether the content achieved SERP traction. This honest boundary is explicitly documented: structural optimization unlocks latent potential, but realized impressions depend heavily on position dynamics.

### Split Leakage Sanity Check (Random vs. Grouped Split)
Evaluating the exact same tuned `HistGradientBoostingRegressor` on a standard random 80/20 train/test split vs. the honest grouped client holdout split:
- **Random Split HGBR $R^2$:** `0.8273` (RMSE `1.1223`, Spearman $\rho = 0.9085$)
- **Grouped Split HGBR $R^2$:** `0.5401` (RMSE `1.8193`, Spearman $\rho = 0.7277$)
- **Artificial Leakage Inflation Gap:** **+0.2872 $R^2$**

**Explanation:** Random splitting leaks client domain authority, root backlinks, and CMS infrastructure between train and test, producing an artificially inflated score. `GroupShuffleSplit` on `client_id` is essential for measuring true multi-tenant generalization.

---

## 4. Operational Refresh Queue

- **Exported File:** [`work/outputs/refresh_queue_sample.csv`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/refresh_queue_sample.csv)
- **Queue Count:** Exactly 50 prioritized candidate URLs.
- **Filtering Threshold:** Every candidate URL strictly meets $\text{Opportunity Residual Gap} = \hat{y} - y > 0.5$ (actual minimum gap is **4.1644**, maximum gap is **6.0449**).
- **Documented Reason Codes:**
  - `stale_authority`: Aged URLs ($> 180$ days old, $> 90$ days since update) with strong authority suffering from content decay.
  - `high_demand_opportunity`: High keyword volume ($> 300$) and solid engagement ($> 20\%$) where an editorial refresh can yield page-1 gains.
  - `under_indexed_decay`: High predicted potential lagging in SERP traction; candidates for structural re-heading and schema enhancement.

---

## 5. Deviations from the Guide & Methodological Enhancements

1. **Exclusion of `days_with_impressions`:** Although not explicitly listed on the blacklist, `days_with_impressions` tracks daily impression presence over the 90-day window and exhibits a $0.917$ Spearman correlation with the target. Including it would constitute target telemetry leakage. It was excluded from the feature matrix to maintain strict scientific honesty.
2. **Handling Constant Baseline Predictions for Spearman Correlation:** When computing Spearman rank correlation for Baseline 1 (Mean Dummy Regressor), predicting a single constant value ($\bar{y}_{\text{train}}$) across all test items results in zero standard deviation in ranks ($0/0 \to \text{NaN}$). This was explicitly handled by mapping constant predictions to `0.0` rather than allowing `null`/`NaN` in JSON metrics.
3. **Encoding and Cross-Platform Execution Safety:** Dynamic project root path resolution was implemented across all three notebooks to allow robust execution from any working directory (`DA-Project/` or `DA-Project/work/notebooks/`), with UTF-8 console reconfiguring to ensure compatibility with Windows PowerShell and continuous integration environments.

---

## 6. Artifact Verification Checklist (All 9 Produced & Validated)

| Artifact Path | Format | Verification Status | Size | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| [`work/outputs/hypothesis_audit_distributions.png`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/hypothesis_audit_distributions.png) | PNG | **PASS** | 367 KB | 2×2 diagnostic distribution hexbin plots |
| [`work/outputs/prepared_features.parquet`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/prepared_features.parquet) | Parquet | **PASS** | 1.62 MB | Engineered, leakage-free feature matrix (49 features) |
| [`work/outputs/prepared_features.csv`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/prepared_features.csv) | CSV | **PASS** | 10.09 MB | Feature vector backup (gitignored) |
| [`work/outputs/model_evaluation_payload.pkl`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/model_evaluation_payload.pkl) | PKL | **PASS** | 915 KB | Test ground truth, model predictions, and split metadata |
| [`work/outputs/model_test_predictions.csv`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/model_test_predictions.csv) | CSV | **PASS** | 916 KB | Row-level predictions for all candidate models |
| [`work/outputs/model_benchmark_metrics.json`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/model_benchmark_metrics.json) | JSON | **PASS** | 1.31 KB | Committed benchmark receipts matching Capstone numbers |
| [`work/outputs/model_evaluation_plots.png`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/model_evaluation_plots.png) | PNG | **PASS** | 328 KB | 1×3 actual-vs-pred, residual KDE, and error boxplots |
| [`work/outputs/refresh_queue_sample.csv`](file:///c:/Users/ACER/Documents/DA-Project/work/outputs/refresh_queue_sample.csv) | CSV | **PASS** | 5.08 KB | Top 50 prioritized refresh queue with reason codes |
| [`work/capstone_report.md`](file:///c:/Users/ACER/Documents/DA-Project/work/capstone_report.md) | Markdown | **PASS** | 19.4 KB | Completed 10-section Capstone research report |

All notebooks ([`01_problem_framing_and_data_audit.ipynb`](file:///c:/Users/ACER/Documents/DA-Project/work/notebooks/01_problem_framing_and_data_audit.ipynb), [`02_model_training.ipynb`](file:///c:/Users/ACER/Documents/DA-Project/work/notebooks/02_model_training.ipynb), [`03_evaluation_and_conclusions.ipynb`](file:///c:/Users/ACER/Documents/DA-Project/work/notebooks/03_evaluation_and_conclusions.ipynb)) and [`work/verify_project.py`](file:///c:/Users/ACER/Documents/DA-Project/work/verify_project.py) are completely executed, self-contained, and verified.

