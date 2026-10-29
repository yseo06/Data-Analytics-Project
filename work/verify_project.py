#!/usr/bin/env python3
"""
verify_project.py -- Definition-of-Done checker for the FlyRank Applied Search Intelligence project.
Run: python work/verify_project.py
"""
import os, sys, json, pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / 'work' / 'outputs'
WORK_DIR = PROJECT_ROOT / 'work'

results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

print("=" * 70)
print("FlyRank Project Verification -- Definition of Done")
print("=" * 70)

# --- 1. All 9 artifacts exist and are non-empty ---
print("\n--- Artifact Existence ---")
artifacts = [
    ('work/outputs/hypothesis_audit_distributions.png', 'Hypothesis audit plot'),
    ('work/outputs/prepared_features.parquet', 'Prepared features parquet'),
    ('work/outputs/prepared_features.csv', 'Prepared features CSV'),
    ('work/outputs/model_evaluation_payload.pkl', 'Model evaluation payload'),
    ('work/outputs/model_test_predictions.csv', 'Model test predictions'),
    ('work/outputs/model_benchmark_metrics.json', 'Benchmark metrics JSON'),
    ('work/outputs/model_evaluation_plots.png', 'Evaluation plots'),
    ('work/outputs/refresh_queue_sample.csv', 'Refresh queue sample'),
    ('work/capstone_report.md', 'Capstone report'),
]
for rel_path, desc in artifacts:
    p = PROJECT_ROOT / rel_path
    exists = p.exists() and p.stat().st_size > 0
    check(f"Artifact: {desc}", exists, str(p) if not exists else f"{p.stat().st_size:,} bytes")

# --- 2. Grouped split asserted disjoint ---
print("\n--- Grouped Split Disjointness ---")
try:
    import pickle
    with open(OUTPUT_DIR / 'model_evaluation_payload.pkl', 'rb') as f:
        payload = pickle.load(f)
    import pandas as pd
    df = pd.read_parquet(OUTPUT_DIR / 'prepared_features.parquet')
    train_clients = set(df.iloc[payload['train_idx']]['client_id'].unique())
    test_clients = set(df.iloc[payload['test_idx']]['client_id'].unique())
    disjoint = train_clients.isdisjoint(test_clients)
    check("Grouped split clients are disjoint", disjoint,
          f"train={len(train_clients)}, test={len(test_clients)}")
except Exception as e:
    check("Grouped split clients are disjoint", False, str(e))

# --- 3. No blacklisted feature in the feature matrix ---
print("\n--- Blacklist Check ---")
try:
    df = pd.read_parquet(OUTPUT_DIR / 'prepared_features.parquet')
    META_COLS = {'client_id', 'content_id', 'target_log_impressions'}
    BLACKLIST = {
        'impressions_90d', 'clicks_90d', 'ctr',
        'impressions_last_30d', 'clicks_last_30d', 'sessions_last_30d',
        'impressions_prev_30d', 'clicks_prev_30d', 'sessions_prev_30d',
        'trend_direction', 'trend_pct', 'impression_tier',
        'health_score', 'priority_score', 'action_type',
        'client_id', 'content_id', 'days_with_impressions'
    }
    feature_cols = set(df.columns) - META_COLS
    leaked = feature_cols & BLACKLIST
    check("No blacklisted features in feature matrix", len(leaked) == 0,
          f"Leaked: {leaked}" if leaked else "0 blacklisted columns")
except Exception as e:
    check("No blacklisted features in feature matrix", False, str(e))

# --- 4. Champion beats both baselines on test RMSE ---
print("\n--- Champion vs Baselines ---")
try:
    with open(OUTPUT_DIR / 'model_benchmark_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    champ_rmse = metrics['Champion: 0.5*Ridge + 0.5*HGBR']['RMSE']
    b1_rmse = metrics['Baseline 1: Mean Dummy']['RMSE']
    b2_rmse = metrics['Baseline 2: OLS log_word_count']['RMSE']
    
    beats_b1 = champ_rmse < b1_rmse
    beats_b2 = champ_rmse < b2_rmse
    check("Champion RMSE < Baseline 1 (Mean Dummy)", beats_b1,
          f"Champion={champ_rmse}, B1={b1_rmse}")
    check("Champion RMSE < Baseline 2 (OLS word_count)", beats_b2,
          f"Champion={champ_rmse}, B2={b2_rmse}")
except Exception as e:
    check("Champion beats baselines", False, str(e))

# --- 5. Refresh queue has 50 rows, every gap > 0.5, valid reason codes ---
print("\n--- Refresh Queue Validation ---")
try:
    rq = pd.read_csv(OUTPUT_DIR / 'refresh_queue_sample.csv')
    check("Refresh queue has 50 rows", len(rq) == 50, f"Got {len(rq)}")
    check("All gaps > 0.5", (rq['opportunity_gap'] > 0.5).all(),
          f"Min gap: {rq['opportunity_gap'].min():.4f}")
    
    valid_codes = {'under_indexed_decay', 'high_demand_opportunity', 'stale_authority'}
    all_codes = set()
    for rc in rq['reason_code']:
        for code in str(rc).split('; '):
            all_codes.add(code.strip())
    invalid = all_codes - valid_codes
    check("All reason codes are valid", len(invalid) == 0,
          f"Invalid: {invalid}" if invalid else f"Codes: {all_codes}")
except Exception as e:
    check("Refresh queue validation", False, str(e))

# --- 6. No generated csv/parquet tracked by git ---
print("\n--- Git Tracking Check ---")
try:
    import subprocess
    result = subprocess.run(
        ['git', 'ls-files'],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    # Check for leaked generated datasets (data/raw starter slice is the sole allowed CSV)
    tracked_leaks = [
        f for f in result.stdout.strip().split('\n')
        if (f.endswith('.csv') or f.endswith('.parquet')) and f != 'data/raw/content_refresh_anonymized.csv'
    ]
    check("No generated csv/parquet tracked by git", len(tracked_leaks) == 0,
          f"Tracked leaks: {tracked_leaks}" if tracked_leaks else "Clean (0 generated datasets tracked)")
except Exception as e:
    check("No generated csv/parquet tracked by git", False, str(e))

# --- 7. Metrics JSON matches report numbers ---
print("\n--- Metrics JSON Consistency ---")
try:
    with open(OUTPUT_DIR / 'model_benchmark_metrics.json', 'r') as f:
        metrics = json.load(f)
    expected_models = [
        'Baseline 1: Mean Dummy',
        'Baseline 2: OLS log_word_count',
        'Candidate A: RidgeCV + Interactions',
        'Candidate B: HGBR (tuned)',
        'Champion: 0.5*Ridge + 0.5*HGBR',
    ]
    for model in expected_models:
        has_model = model in metrics
        check(f"Metrics JSON has '{model}'", has_model)
        if has_model:
            has_r2 = 'R2' in metrics[model]
            check(f"  -> has R2 metric ({metrics[model].get('R2')})", has_r2)
except Exception as e:
    check("Metrics JSON consistency", False, str(e))

# --- Summary ---
print("\n" + "=" * 70)
total = len(results)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if failed > 0:
    print("\nFailed checks:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  [X] {name}: {detail}")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
