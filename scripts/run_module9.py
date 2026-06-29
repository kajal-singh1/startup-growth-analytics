"""
run_module9.py — Run Hyperparameter Tuning
=======================================================================
Usage:
    python scripts/run_module9.py
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
    print("  MODULE 9 — HYPERPARAMETER TUNING")
    print("=" * 60)

    from tuning.module9_hyperparameter_tuning import run as run_m9
    leaderboard_df, best_xgb, best_rf = run_m9(config)

    print("\n" + "=" * 60)
    print("  MODULE 9 COMPLETE")
    print("=" * 60)
    print(f"  Overall winner: {leaderboard_df.iloc[0]['model']}  "
          f"(Test R²={leaderboard_df.iloc[0]['test_r2']:.4f})")
    print("  models/tuned/module9_xgboost_tuned.joblib       ✓")
    print("  models/tuned/module9_random_forest_tuned.joblib ✓")
    print("  data/processed/module9_final_leaderboard.csv    ✓")
    print("  outputs/figures/ — 10 new figures                ✓")
    print("  outputs/reports/ — 2 new reports                  ✓")
    print()
    print("  Next: Module 11 (Explainable AI)")
    print("=" * 60)


if __name__ == "__main__":
    main()
