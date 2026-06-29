"""
run_module7.py — Run Module 7: Statistical Analysis
=====================================================
Usage:
    python scripts/run_module7.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config, set_seeds
from stats.module7_statistical_analysis import run as run_stats


def main():
    config = load_config()
    set_seeds(config)

    print("\n" + "="*60)
    print("  MODULE 7: STATISTICAL ANALYSIS")
    print("="*60)

    df, corr_df, result, tests = run_stats(config)

    print("\n" + "="*60)
    print("  MODULE 7 COMPLETE")
    print("="*60)
    print("  outputs/figures/module7_01 … 10   ✓  (10 figures)")
    print("  outputs/reports/module7_stats_report ✓")
    print()
    print("  Next: python scripts\\run_module8.py  (Machine Learning)")
    print("="*60)


if __name__ == "__main__":
    main()
