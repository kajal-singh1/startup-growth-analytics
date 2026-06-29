"""
run_module8.py — Run Machine Learning
=======================================================================
Usage:
    python scripts/run_module8.py
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

    print("\n" + "=" * 60)
    print("  MODULE 8 — MACHINE LEARNING")
    print("=" * 60)

    from ml.module8_machine_learning import run as run_m8
    best_model, results_df = run_m8(config)

    print("\n" + "=" * 60)
    print("  MODULE 8 COMPLETE")
    print("=" * 60)
    print(f"  Best model: {results_df.iloc[0]['model']}  "
          f"(Test R²={results_df.iloc[0]['test_r2']:.4f})")
    print("  models/saved/module8_best_model.joblib       ✓")
    print("  data/processed/module8_model_leaderboard.csv ✓")
    print("  outputs/figures/ — 10 new figures            ✓")
    print("  outputs/reports/ — 2 new reports              ✓")
    print()
    print("  Next: Module 9 (Hyperparameter Tuning)")
    print("=" * 60)


if __name__ == "__main__":
    main()
