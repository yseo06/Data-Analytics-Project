# FlyRank Applied Search Intelligence: Full Project Guide & Phased Roadmap

> **Document Status:** Complete Project Blueprint & Step-by-Step Execution Manual  
> **Repository:** [DA-Project](file:///c:/Users/ACER/Documents/DA-Project)  
> **Track:** Applied Machine Learning Internship — Applied Search Intelligence (Google Search Ranking & Discoverability)  
> **Target Audience:** Intern / Applied ML Engineer  

---

## Table of Contents
1. [Project Overview & Core Mission](#1-project-overview--core-mission)
2. [Repository Directory & File Architecture](#2-repository-directory--file-architecture)
3. [The Golden Rules: Data Safety, Integrity & Anti-Leakage](#3-the-golden-rules-data-safety-integrity--anti-leakage)
4. [Master Phasing & Step-by-Step Instructions](#4-master-phasing--step-by-step-instructions)
   - [Phase 0: Environment Verification & Foundation Setup](#phase-0-environment-verification--foundation-setup)
   - [Phase 1: Problem Framing, Statistical SEO Hypotheses & Data Ingestion (Notebook 01)](#phase-1-problem-framing-statistical-seo-hypotheses--data-ingestion-notebook-01)
   - [Phase 2: Grouped Holdout Split, Heuristic Baselines & Model Training (Notebook 02)](#phase-2-grouped-holdout-split-heuristic-baselines--model-training-notebook-02)
   - [Phase 3: Benchmark Comparisons, Diagnostics & Operational Action Playbook (Notebook 03)](#phase-3-benchmark-comparisons-diagnostics--operational-action-playbook-notebook-03)
   - [Phase 4: Capstone Report Synthesis & Portfolio Packaging](#phase-4-capstone-report-synthesis--portfolio-packaging)
5. [Artifacts & Outputs Tracking Checklist](#5-artifacts--outputs-tracking-checklist)
6. [Summary Execution Quick-Sheet](#6-summary-execution-quick-sheet)

---

## 1. Project Overview & Core Mission

### 1.1 The Business & Operational Problem
Enterprise content and search optimization teams manage massive portfolios of URLs across diverse client domains. Currently, content refresh prioritization relies on crude, lagging manual heuristics:
* Sorting purely by calendar age (refreshing old pages regardless of whether they have traffic potential).
* Sorting by raw past pageviews (ignoring structural decay or unrealized search demand).
* Subjective editorial guesswork.

This project builds an **Applied Machine Learning Decision-Support System** that predicts expected **Continuous Search Visibility** ($\log(1 + \text{impressions\_90d})$) based strictly on pre-ranking structural, semantic, and engagement signals.

### 1.2 How Teams Act on the Output
The model’s predictions are translated into an **Automated Refresh Triage Queue** by calculating the **Opportunity Residual Gap**:
$$\text{Opportunity Residual Gap} = \hat{y}_{\text{pred}} - y_{\text{actual}}$$

* **Under-Indexed High-Potential Pages ($\hat{y} \gg y$):** Pages whose structural depth, keyword relevance, and engagement warrant high search impressions, but whose realized traffic is lagging. These are the **#1 prime candidates for content updates and internal link boosts**.
* **Over-Resourced Low-Potential Pages ($\hat{y} \ll y$):** Thin or misplaced content that should not consume expensive subject-matter expert rewrite hours.

### 1.3 The Core Workflow Philosophy
```text
Problem Framing → Data Ingestion & Audit → Leakage Defense → Heuristic Baseline → Honest ML Model → Error Diagnostics → Operational Refresh Queue → Capstone Paper
```

---

## 2. Repository Directory & File Architecture

```text
DA-Project/
├── FULL_PROJECT_GUIDE.md                <-- This master operating manual
├── README.md                            <-- Starter repo front door & quickstart
├── GUIDE.md                             <-- Technical architecture guide
├── SETUP.md                             <-- Environment, Colab, & Hugging Face setup
├── DATA_USE.md                          <-- Mandatory data usage and compliance rules
├── requirements.txt                     <-- Python dependency specifications
│
├── data/
│   ├── raw/
│   │   └── content_refresh_anonymized.csv  <-- 30,000 rows × 44 cols (32 clients). READ-ONLY!
│   └── processed/                       <-- Generated intermediate datasets (gitignored)
│
├── docs/
│   ├── data-dictionary.md               <-- Full breakdown of all 44 raw columns (KEEP OPEN)
│   ├── ml-intern-dataset-and-lane-guide.md <-- Deep guide to lanes and Hugging Face warehouse
│   ├── ml-core-foundation-framework.md  <-- ML systems foundations and mental models
│   └── intern-free-tooling-guide.md     <-- Zero-budget tool stack documentation
│
├── skills/                              <-- AI pair-programming skills library
│   ├── README.md                        <-- The skill router
│   ├── flyrank/flyrank-data/SKILL.md    <-- Dataset specifics & gotchas (read for all data work)
│   ├── auditing-signals/SKILL.md        <-- EDA & signal correlation checks
│   ├── building-baselines/SKILL.md      <-- Heuristic baseline formulation
│   ├── training-honest-models/SKILL.md  <-- Leakage-free training & model tuning
│   └── hunting-leakage-and-validating/  <-- Residual checks & holdout validation
│
└── work/                                <-- YOUR WORKSPACE (Where all your deliverables live)
    ├── README.md                        <-- Rules of the road for work/
    ├── capstone_report_template.md      <-- 10-section Capstone rubric template
    ├── create_template_notebooks.py     <-- Script that created the 3 modular notebooks
    ├── notebooks/
    │   ├── 01_problem_framing_and_data_audit.ipynb
    │   ├── 02_model_training.ipynb
    │   └── 03_evaluation_and_conclusions.ipynb
    └── outputs/                         <-- Saved figures, metric JSONs, and sample CSVs
```

---

## 3. The Golden Rules: Data Safety, Integrity & Anti-Leakage

Before writing a single line of code, understand these strict repository guardrails:

### 3.1 Data Safety & Compliance
1. **Never commit dataset files:** Raw or processed `.csv` / `.parquet` files must never be pushed to Git. Continuous Integration (`.github/workflows/smoke-test.yml`) will fail any commit containing a dataset.
2. **Never expose client identities or raw queries:** All client IDs (`client_id`) and content IDs (`content_id`) are pseudonyms. Do not reverse-engineer or expose client names.
3. **Public-Safety Language:** Frame all results as **observed**, **measured**, **directional**, and **decision-support**. Never write *"I predicted Google's ranking algorithm"* (an algorithm with thousands of unobservable signals cannot be claimed as "solved").

### 3.2 Anti-Leakage Golden Rules
1. **Target Telemetry Leakage (Blacklist):**
   * Target variable is continuous search visibility: $\text{target\_log\_impressions} = \log(1 + \text{impressions\_90d})$.
   * Therefore, `impressions_90d`, `clicks_90d`, and `ctr` ($\text{clicks} / \text{impressions}$) **must NEVER be features**.
   * Comparative window telemetry (`*_last_30d`, `*_prev_30d`) and target-derived labels (`trend_direction`, `trend_pct`, `impression_tier`) **must NEVER be features**.
   * Internal FlyRank product scores (`health_score`, `priority_score`, `action_type`) are decisions made after the fact and **must NEVER be features**.
2. **Multi-Tenant Client Domain Leakage (Crucial Split Rule):**
   * Pages from the same client share identical domain authority, CMS structure, backlink profiles, and brand trust.
   * **Never use random `train_test_split`.** Doing so leaks client-level domain authority between train and test, producing artificially inflated scores that fail in production.
   * **Always use `GroupShuffleSplit` on `client_id`** with programmatic disjointness assertion:
     $$\text{set}(\text{clients}_{\text{train}}) \cap \text{set}(\text{clients}_{\text{test}}) = \emptyset$$

### 3.3 Data Gotchas
* **Rate columns are $\times 100$ percentages:** In `ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, and `trend_pct`, a value of `0.76` means **0.76%**, not 76%.
* **`avg_position = 0` denotes unranked:** 1,205 rows have average position 0. This means no impression position was recorded (unranked), not page 1 position 0.
* **Systematic Missingness:** Missingness follows `content_type` lines (e.g., `feedly article` rows have zero keyword data; ~25.7% of rows lack `word_count`). **Do not blindly do `fillna(0)`**—create missingness indicator flags.

---

## 4. Master Phasing & Step-by-Step Instructions

---

### Phase 0: Environment Verification & Foundation Setup

**Objective:** Ensure dependencies, data paths, and notebook templates are clean and ready.

#### Detailed Steps:
1. **Activate Environment & Install Dependencies:**
   ```powershell
   cd c:\Users\ACER\Documents\DA-Project
   pip install -r requirements.txt
   ```
2. **Verify Raw Data Presence:**
   Check that [`data/raw/content_refresh_anonymized.csv`](file:///c:/Users/ACER/Documents/DA-Project/data/raw/content_refresh_anonymized.csv) exists (30,000 rows × 44 columns, ~6.75 MB).
3. **Verify/Generate Modular Notebooks:**
   Run the notebook generator if the templates need resetting:
   ```powershell
   python work/create_template_notebooks.py
   ```
   This generates:
   * `work/notebooks/01_problem_framing_and_data_audit.ipynb`
   * `work/notebooks/02_model_training.ipynb`
   * `work/notebooks/03_evaluation_and_conclusions.ipynb`
4. **Create Outputs Directory:**
   Ensure `work/outputs/` exists for saving artifacts:
   ```powershell
   New-Item -ItemType Directory -Force -Path work\outputs
   ```

---

### Phase 1: Problem Framing, Statistical SEO Hypotheses & Data Ingestion (Notebook 01)

📁 **Working File:** [`work/notebooks/01_problem_framing_and_data_audit.ipynb`](file:///c:/Users/ACER/Documents/DA-Project/work/notebooks/01_problem_framing_and_data_audit.ipynb)  
📖 **Skills to Reference:** `skills/auditing-signals/SKILL.md` and `skills/flyrank/flyrank-data/SKILL.md`

#### Step 1.1: Data Ingestion & Entity Grain Assertion
* Ingest `data/raw/content_refresh_anonymized.csv`.
* Verify the dataset grain: Assert exactly 1 row per unique `content_id` ($N = 30,000$).
* Count unique enterprise clients: Assert exactly 32 distinct `client_id`s.
* Compute and append the continuous regression target:
  $$\text{target\_log\_impressions} = \log(1 + \text{impressions\_90d})$$
* Check and display descriptive statistics of `target_log_impressions` (mean, median, standard deviation, skewness).

#### Step 1.2: Completeness & Systematic Missingness Audit
* Calculate missing value counts and percentages for every column.
* Document key missingness patterns:
  * `word_count` and `char_count`: Missing for ~25.7% of rows ($N = 7,699$).
  * Keyword metrics (`search_volume`, `competition`, `competition_level`, `cpc`): Missing for $N = 2,468$ rows (systematic along `content_type == 'feedly article'`).
  * `avg_position`: Identify the 1,205 unranked rows where `avg_position == 0`.

#### Step 1.3: Statistical SEO Hypothesis Testing
Formulate and test the 4 core search hypotheses against `target_log_impressions`. Compute both **Spearman Rank Correlation ($\rho$)** (monotonic) and **Pearson Correlation ($r$)** (linear), assigning explicit statistical verdicts:
1. **Hypothesis 1 (Content Depth):** Article length (`word_count`) correlates positively with search visibility.
   * *Expected Verdict:* `CONFIRMED` / moderate positive correlation.
2. **Hypothesis 2 (Search Demand):** Keyword volume (`search_volume`) strongly determines realized page impressions.
   * *Expected Verdict:* `MIXED` (Broad head terms face severe SERP crowding; volume alone does not guarantee impressions).
3. **Hypothesis 3 (SERP Positioning):** Rank position (`avg_position`) is inversely correlated with impressions (evaluated on ranked rows `avg_position > 0`).
   * *Expected Verdict:* `FALSE` / weak in isolation (High rank on low-volume terms yields fewer impressions than page 2 on massive head terms).
4. **Hypothesis 4 (Content Freshness):** Recently updated content (`days_since_last_update` lower) commands higher impressions.
   * *Expected Verdict:* `OPPOSITE` (Mature, aging content with accumulated domain authority and backlinks often outperforms newly refreshed thin content).
* Save the 2×2 diagnostic hypothesis distribution plot to `work/outputs/hypothesis_audit_distributions.png`.

#### Step 1.4: Domain-Specific Feature Engineering
Construct clean, high-leverage predictive signals without data leakage:
1. **SERP Power-Law Inverse Rank:**
   $$\text{serp\_rank\_power} = \begin{cases} \frac{1}{\text{avg\_position}}, & \text{if } \text{avg\_position} > 0 \\ 0, & \text{otherwise} \end{cases}$$
2. **Rank Status Indicators & Imputation:**
   * `has_rank_data = (avg_position > 0).astype(int)`
   * `imputed_avg_position = np.where(avg_position > 0, avg_position, 100.0)`
3. **Behavioral Engagement Ratios:**
   * $\text{engaged\_session\_ratio} = \frac{\text{engaged\_sessions\_90d}}{\text{sessions\_90d} + 1}$
   * $\text{scroll\_per\_pageview} = \frac{\text{scroll\_events\_90d}}{\text{pageviews\_90d} + 1}$
   * $\text{ai\_session\_ratio} = \frac{\text{ai\_sessions\_90d}}{\text{sessions\_90d} + 1}$
4. **Synergy Interaction Terms:**
   * $\text{demand\_position\_synergy} = \log(1 + \text{search\_volume}) \times \text{serp\_rank\_power}$
   * $\text{engagement\_volume\_synergy} = \log(1 + \text{sessions\_90d}) \times \text{serp\_rank\_power}$
5. **Systematic Missingness Indicator Flags:**
   * `missing_word_count`, `missing_search_volume`, `missing_competition`, `missing_cpc`
6. **Log-Stabilizing Transformations:**
   * Apply `np.log1p` to heavy-tailed count columns (`word_count`, `search_volume`, `sessions_90d`, `pageviews_90d`, `users_90d`, `content_age_days`, `days_since_last_update`).
7. **Categorical Encoding:**
   * Apply one-hot encoding (`pd.get_dummies`) to `content_type`, `main_intent`, `competition_level`, `freshness_tier`, `position_tier`.

#### Step 1.5: Strict Anti-Leakage Defense & Dataset Export
* Enforce blacklist assertions: Confirm zero presence of target variables or label derivatives (`impressions_90d`, `clicks_90d`, `ctr`, `trend_direction`, `trend_pct`, `impression_tier`, outcome 30d metrics).
* Export the clean feature vector:
  * `work/outputs/prepared_features.parquet`
  * `work/outputs/prepared_features.csv`

---

### Phase 2: Grouped Holdout Split, Heuristic Baselines & Model Training (Notebook 02)

📁 **Working File:** [`work/notebooks/02_model_training.ipynb`](file:///c:/Users/ACER/Documents/DA-Project/work/notebooks/02_model_training.ipynb)  
📖 **Skills to Reference:** `skills/building-baselines/SKILL.md` and `skills/training-honest-models/SKILL.md`

#### Step 2.1: Grouped Holdout Split (Zero Client Domain Leakage)
* Load `work/outputs/prepared_features.parquet`.
* Separate features $X$, target $y$ (`target_log_impressions`), and groups (`client_id`).
* Execute `GroupShuffleSplit`:
  ```python
  from sklearn.model_selection import GroupShuffleSplit
  gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
  train_idx, test_idx = next(gss.split(X, y, groups=df['client_id']))
  ```
* Programmatically assert zero client domain leakage:
  ```python
  assert set(df.loc[train_idx, 'client_id']).isdisjoint(set(df.loc[test_idx, 'client_id']))
  ```
  *(Result: 25 training clients / ~23,837 rows vs. 7 held-out test clients / ~6,163 rows).*

#### Step 2.2: Establish Transparent Heuristic Baselines
Before fitting any ML model, establish the performance floor on the exact same test cohort:
1. **Baseline 1 — Mean Dummy Regressor:**
   * Always predicts the mean of the training target ($\bar{y}_{\text{train}}$).
   * Establishes the naive benchmark ($R^2 = 0.0$ on train; often negative on held-out test clients).
2. **Baseline 2 — Univariate Word Count Rule:**
   * Single-variable Ordinary Least Squares (OLS) regression on `log_word_count`.
   * Directly tests whether simple content length alone explains search visibility.
* Compute baseline metrics: $R^2$, RMSE, MAE, and Spearman $\rho$.

#### Step 2.3: Candidate Machine Learning Model Training
Fit and optimize candidate models designed for multi-tenant generalization:
1. **Candidate A: Regularized Polynomial Linear Model (`RidgeCV`)**
   * Impute missing features (`SimpleImputer(strategy='median')`), standardize (`StandardScaler`), add pairwise interactions on core signals, and fit cross-validated L2 regularization.
2. **Candidate B: Gradient Boosted Decision Trees (`HistGradientBoostingRegressor`)**
   * Natively handles missing values and non-linear step thresholds without requiring imputation.
   * Tune `learning_rate`, `max_iter`, `max_leaf_nodes`, and `l2_regularization`.
3. **Candidate C: Champion Blended Ensemble**
   * Combine linear macro-trend generalization with non-linear tree splits:
     $$\hat{y}_{\text{Champion}} = 0.50 \cdot \hat{y}_{\text{Ridge}} + 0.50 \cdot \hat{y}_{\text{HGBR}}$$

#### Step 2.4: Payload Packaging & Export
* Evaluate all candidates on the unseen test client set.
* Package ground truth, predictions, client identifiers, and metadata into a clean evaluation payload.
* Export files:
  * `work/outputs/model_evaluation_payload.pkl`
  * `work/outputs/model_test_predictions.csv`

---

### Phase 3: Benchmark Comparisons, Diagnostics & Operational Action Playbook (Notebook 03)

📁 **Working File:** [`work/notebooks/03_evaluation_and_conclusions.ipynb`](file:///c:/Users/ACER/Documents/DA-Project/work/notebooks/03_evaluation_and_conclusions.ipynb)  
📖 **Skills to Reference:** `skills/hunting-leakage-and-validating/SKILL.md`

#### Step 3.1: Official Benchmark Performance Table
* Load `work/outputs/model_evaluation_payload.pkl`.
* Compile the official benchmark comparison table across:
  * **$R^2$ Score (Variance Explained):** Did the model beat the negative/zero baseline on unseen enterprise domains?
  * **RMSE & MAE:** Error magnitude in log-impression units.
  * **Percentage RMSE Reduction:** $\frac{\text{RMSE}_{\text{Baseline}} - \text{RMSE}_{\text{Model}}}{\text{RMSE}_{\text{Baseline}}} \times 100\%$.
  * **Spearman Rank Correlation ($\rho$):** Monotonic ordering accuracy for ranking queues.
* Export metrics to `work/outputs/model_benchmark_metrics.json`.

#### Step 3.2: Diagnostic Error Visualizations
Generate and save a 1×3 diagnostic figure (`work/outputs/model_evaluation_plots.png`):
1. **Plot 1: Actual vs. Predicted Scatter:** Showing test cohort points against the 45-degree identity line ($y = \hat{y}$).
2. **Plot 2: Residual Distribution:** Error histogram ($y_{\text{pred}} - y_{\text{actual}}$) with Kernel Density Estimation (KDE) and zero-bias line.
3. **Plot 3: Absolute Error Boxplot:** Comparing distribution of errors across Baseline 1, Baseline 2, and the Champion Model.

#### Step 3.3: Sub-Cohort Robustness Audit
* Segment test client performance by `content_type` (`keyword article`, `feedly article`, `comparison article`) and `main_intent` (`informational`, `transactional`, `commercial`).
* Confirm that $R^2$ and Spearman $\rho$ remain stable across segments rather than collapsing on specific content types.

#### Step 3.4: In-Depth Outlier Case Studies (Post-Mortem)
Select and analyze 3 distinct failure/outlier cases from the test set:
* **Case 1 (Under-Predicted High Performer):** Low predicted visibility, but high actual impressions. *Diagnosis:* Viral social/backlink spike or branded navigational query not captured by pre-ranking features.
* **Case 2 (Over-Predicted Low Performer):** High predicted visibility, but low actual impressions. *Diagnosis:* High keyword search volume, but extreme SERP competition or click cannibalization from Google AI Overviews.
* **Case 3 (Prime Refresh Opportunity):** High predicted visibility, lagging realized impressions, high page age. *Diagnosis:* Strong latent demand and topical authority suffering from decay—ideal refresh candidate.

#### Step 3.5: Build the Automated Content Refresh Queue
Translate model predictions into operational business actions:
1. Compute the **Opportunity Residual Gap**: $\text{opportunity\_gap} = \hat{y}_{\text{log\_impressions}} - y_{\text{log\_impressions}}$.
2. Transform back to raw scale: $\text{pred\_impressions} = \exp(\hat{y}) - 1$.
3. Filter for pages with strong latent potential ($\text{opportunity\_gap} > 0.5$).
4. Assign actionable **Reason Codes**:
   * `under_indexed_decay`: High predicted potential lagging in realized impressions.
   * `high_demand_opportunity`: High target search volume with solid engagement warranting a SERP push.
   * `stale_authority`: Aged high-authority URL needing updated data/statistics.
5. Rank by priority and export the top 50 refresh candidates to `work/outputs/refresh_queue_sample.csv`.

---

### Phase 4: Capstone Report Synthesis & Portfolio Packaging

📁 **Working File:** Create `work/capstone_report.md` from [`work/capstone_report_template.md`](file:///c:/Users/ACER/Documents/DA-Project/work/capstone_report_template.md)

#### Step 4.1: Fill in the Capstone Report (10 Required Sections)
* **0. Abstract:** 5 sentences (Question $\rightarrow$ Data $\rightarrow$ Method $\rightarrow$ Headline Result $\rightarrow$ Operational Action).
* **1. Problem Framing:** Unit of analysis, target definition, operational action, cost of asymmetric errors.
* **2. Data Safety & Integrity:** Blacklisted features, rate column adjustments, missingness indicator strategy, assertion of zero client leakage.
* **3. Baseline:** Formulation and metrics of Mean Dummy and Word Count Rule.
* **4. Model Architecture:** Regularized Ridge, HistGradientBoosting, and Ensemble formulation; feature list.
* **5. Evaluation & Validation:** Grouped Holdout split metrics table ($R^2$, RMSE, MAE, Spearman $\rho$).
* **6. Interpretation:** Top feature importances (e.g., SERP rank power, engagement synergy) and negative/surprising results.
* **7. Operational Recommendation:** How editorial teams use the refresh queue and reason codes tomorrow; limitations.
* **8. Reproducibility:** Exact commands, seed (`random_state=42`), library versions.
* **9. Acknowledgments & Data Credit:** "Built on the FlyRank ML Internship dataset" with link to `https://flyrank.ai`.

#### Step 4.2: End-to-End Validation
* In Jupyter, restart each notebook kernel and execute **Run All Cells** for:
  1. `01_problem_framing_and_data_audit.ipynb`
  2. `02_model_training.ipynb`
  3. `03_evaluation_and_conclusions.ipynb`
* Confirm all outputs in `work/outputs/` match the report figures and tables exactly.

---

## 5. Artifacts & Outputs Tracking Checklist

Ensure the following files are produced in `work/outputs/` and properly tracked:

| Artifact Path | Format | Generated By | Purpose |
| :--- | :--- | :--- | :--- |
| `work/outputs/hypothesis_audit_distributions.png` | PNG image | Notebook 01 | 2×2 grid of SEO hypothesis scatter distributions |
| `work/outputs/prepared_features.parquet` | Parquet | Notebook 01 | Clean, engineered feature vector matrix |
| `work/outputs/prepared_features.csv` | CSV | Notebook 01 | Feature vector backup (gitignored) |
| `work/outputs/model_evaluation_payload.pkl` | Pickle | Notebook 02 | Test ground truth, predictions, and metadata payload |
| `work/outputs/model_test_predictions.csv` | CSV | Notebook 02 | Row-level test predictions table (gitignored) |
| `work/outputs/model_benchmark_metrics.json` | JSON | Notebook 03 | Official benchmark scores (committed receipt) |
| `work/outputs/model_evaluation_plots.png` | PNG image | Notebook 03 | 1×3 diagnostic evaluation plots |
| `work/outputs/refresh_queue_sample.csv` | CSV | Notebook 03 | Prioritized top 50 refresh queue with reason codes |
| `work/capstone_report.md` | Markdown | Phase 4 | Completed capstone research report |

---

## 6. Summary Execution Quick-Sheet

```powershell
# 1. Environment & Setup
pip install -r requirements.txt
python work/create_template_notebooks.py

# 2. Launch Notebook Server
jupyter notebook work/notebooks/

# 3. Complete Notebook 01: Audit, Hypotheses & Features
#    -> Generates prepared_features.parquet and hypothesis_audit_distributions.png

# 4. Complete Notebook 02: Grouped Holdout & Model Training
#    -> Generates model_evaluation_payload.pkl

# 5. Complete Notebook 03: Diagnostics & Refresh Queue
#    -> Generates model_benchmark_metrics.json, evaluation plots, and refresh_queue_sample.csv

# 6. Finalize Capstone
#    -> Copy work/capstone_report_template.md to work/capstone_report.md and fill in results
```
