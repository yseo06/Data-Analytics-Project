#!/usr/bin/env python3
"""
run_all_notebooks.py — Extracts and executes all code cells from the three notebooks sequentially.
This is equivalent to running all three notebooks top-to-bottom with a fresh kernel.
"""
import json, sys, os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))

def run_notebook(nb_path):
    """Extract code cells from a notebook and execute them in sequence."""
    print(f"\n{'='*70}")
    print(f"EXECUTING: {nb_path}")
    print(f"{'='*70}")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Collect all code cells
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    combined_code = ""
    for i, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        combined_code += f"\n# --- Cell {i+1} ---\n{source}\n"
    
    # Execute all code
    exec(compile(combined_code, nb_path, 'exec'), {'__name__': '__main__'})
    print(f"\n✓ {nb_path} completed successfully.")

if __name__ == '__main__':
    notebooks = [
        'notebooks/01_problem_framing_and_data_audit.ipynb',
        'notebooks/02_model_training.ipynb',
        'notebooks/03_evaluation_and_conclusions.ipynb',
    ]
    for nb in notebooks:
        run_notebook(nb)
    print(f"\n{'='*70}")
    print("ALL NOTEBOOKS EXECUTED SUCCESSFULLY")
    print(f"{'='*70}")

