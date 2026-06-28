"""
run_module6.py — Run Module 6: EDA
====================================
Usage:
    python scripts/run_module6.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config, set_seeds
from eda.module6_eda import run as run_eda


def main():
    config = load_config()
    set_seeds(config)

    print("\n" + "="*60)
    print("  MODULE 6: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    df, stats = run_eda(config)

    print("\n" + "="*60)
    print("  MODULE 6 COMPLETE")
    print("="*60)
    print("  outputs/figures/module6_01 … 12   ✓  (12 figures)")
    print("  outputs/reports/module6_eda_report ✓")
    print()
    print("  Next: python scripts\\run_module7.py  (Statistical Analysis)")
    print("="*60)


if __name__ == "__main__":
    main()
