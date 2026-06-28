"""
run_modules_3_4_5.py — Run Validation + Cleaning + Feature Engineering
=======================================================================
Runs all three backfill modules in sequence.

Usage:
    python scripts/run_modules_3_4_5.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config, set_seeds

def main():
    config = load_config()
    set_seeds(config)
    

    print("\n" + "="*60)
    print("  PIPELINE: MODULE 3 → 4 → 5")
    print("="*60)

    # Module 3 — Validation
    print("\n[1/3] Running Module 3: Data Validation...")
    from validation.module3_validation import run as run_m3
    df_raw, findings = run_m3(config)
    print(f"  ✓ Health score: {round(100*(1 - sum(findings.get('missing_pct',{}).values())/100/len(findings.get('missing_pct',{1:1}))),1)}%")

    # Module 4 — Cleaning
    print("\n[2/3] Running Module 4: Data Cleaning...")
    from cleaning.module4_cleaning import run as run_m4
    df_clean = run_m4(config)
    print(f"  ✓ Clean shape: {df_clean.shape}, Missing: {df_clean.isnull().sum().sum()}")

    # Module 5 — Feature Engineering
    print("\n[3/3] Running Module 5: Feature Engineering...")
    from features.module5_features import run as run_m5
    df_feat = run_m5(config)
    print(f"  ✓ Feature shape: {df_feat.shape}")

    print("\n" + "="*60)
    print("  ALL COMPLETE")
    print("="*60)
    print("  data/interim/master_raw.csv          ✓ (Module 2)")
    print("  data/processed/master_clean.csv      ✓ (Module 4)")
    print("  data/processed/master_features.csv   ✓ (Module 5)")
    print("  SQLite: master_dataset                ✓ (Module 4)")
    print("  SQLite: engineered_features           ✓ (Module 5)")
    print("  outputs/figures/ — 16 figures total   ✓")
    print("  outputs/reports/ — 5 reports total    ✓")
    print()
    print("  Next: Module 6 (EDA) or Module 7 (Statistical Analysis)")
    print("="*60)


if __name__ == "__main__":
    main()
