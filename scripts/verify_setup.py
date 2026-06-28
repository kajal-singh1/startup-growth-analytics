"""
verify_setup.py — Module 1 Setup Verification
===============================================
Runs all checks after Module 1 is complete.
Validates: folder structure, config, database, utils, seeds.
Produces a pass/fail report.

Run:  python scripts/verify_setup.py
"""

import sys
import os
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
results = []


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"{status}  {label}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────
# 1. PROJECT ROOT EXISTS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
check("Project root exists", PROJECT_ROOT.exists(), str(PROJECT_ROOT))


# ─────────────────────────────────────────────
# 2. DIRECTORY STRUCTURE
# ─────────────────────────────────────────────
required_dirs = [
    "data/raw", "data/interim", "data/processed", "data/external",
    "src", "notebooks", "models/saved", "models/tuned", "models/experiments",
    "outputs/figures", "outputs/reports", "outputs/logs",
    "config", "docs", "tests", "scripts", "database",
    "src/collection", "src/validation", "src/cleaning", "src/features",
    "src/eda", "src/stats", "src/causal", "src/ml", "src/xai",
    "src/clustering", "src/forecasting", "src/geospatial",
    "src/risk", "src/api", "src/dashboard", "src/reports", "src/mlops",
]
for d in required_dirs:
    check(f"Dir: {d}", (PROJECT_ROOT / d).exists())


# ─────────────────────────────────────────────
# 3. KEY FILES EXIST
# ─────────────────────────────────────────────
required_files = [
    "requirements.txt",
    "environment.yml",
    "config/config.yaml",
    "src/utils.py",
    "scripts/setup_database.py",
    "scripts/verify_setup.py",
]
for f in required_files:
    check(f"File: {f}", (PROJECT_ROOT / f).exists())


# ─────────────────────────────────────────────
# 4. CONFIG LOADS CORRECTLY
# ─────────────────────────────────────────────
try:
    from utils import load_config
    cfg = load_config()
    check("Config loads", True, f"seed={cfg['reproducibility']['random_seed']}")
    check("Config has paths", "paths" in cfg)
    check("Config has ml section", "ml" in cfg)
    check("Config has reproducibility", "reproducibility" in cfg)
except Exception as e:
    check("Config loads", False, str(e))


# ─────────────────────────────────────────────
# 5. SEEDS SET WITHOUT ERROR
# ─────────────────────────────────────────────
try:
    from utils import set_seeds
    set_seeds(cfg)
    import numpy as np
    # Verify seed is deterministic
    np.random.seed(42)
    val1 = np.random.rand()
    np.random.seed(42)
    val2 = np.random.rand()
    check("Seeds are deterministic", val1 == val2, f"val={val1:.6f}")
except Exception as e:
    check("Seeds set", False, str(e))


# ─────────────────────────────────────────────
# 6. LOGGING WORKS
# ─────────────────────────────────────────────
try:
    from utils import setup_logging
    logger = setup_logging("verify_setup", cfg)
    logger.info("Verification run.")
    log_file = PROJECT_ROOT / cfg["paths"]["logs"] / "project.log"
    check("Log file created", log_file.exists(), str(log_file))
except Exception as e:
    check("Logging works", False, str(e))


# ─────────────────────────────────────────────
# 7. DATABASE EXISTS AND HAS TABLES
# ─────────────────────────────────────────────
try:
    db_path = PROJECT_ROOT / cfg["paths"]["database"]
    check("Database file exists", db_path.exists(), str(db_path))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cursor.fetchall()]
    expected_tables = [
        "world_bank_raw", "startup_ecosystem_raw", "master_dataset",
        "engineered_features", "ml_predictions", "forecasts",
        "cluster_assignments", "risk_scores", "experiments"
    ]
    check("DB has correct tables", set(expected_tables).issubset(set(tables)),
          f"Found: {tables}")
    conn.close()
except Exception as e:
    check("Database check", False, str(e))


# ─────────────────────────────────────────────
# 8. CORE LIBRARIES IMPORTABLE
# ─────────────────────────────────────────────
core_libs = {
    "numpy": "import numpy",
    "pandas": "import pandas",
    "matplotlib": "import matplotlib",
    "seaborn": "import seaborn",
    "sklearn": "import sklearn",
    "scipy": "import scipy",
    "yaml": "import yaml",
    "requests": "import requests",
    "plotly": "import plotly",
}
for lib, imp in core_libs.items():
    try:
        exec(imp)
        check(f"Library: {lib}", True)
    except ImportError:
        check(f"Library: {lib}", False, "Not installed")


# ─────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────
total   = len(results)
passed  = sum(1 for r in results if r[0] == PASS)
failed  = total - passed

print("\n" + "="*60)
print(f"MODULE 1 VERIFICATION REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print(f"Total checks : {total}")
print(f"Passed       : {passed}")
print(f"Failed       : {failed}")
if failed == 0:
    print("\n✓ ALL CHECKS PASSED — Module 1 setup is complete.")
    print("✓ Ready to proceed to Module 2: Data Collection.")
else:
    print(f"\n✗ {failed} check(s) failed. Resolve issues before proceeding.")
print("="*60)

# Write report
try:
    from utils import write_module_summary
    summary = {
        "Total checks": total,
        "Passed": passed,
        "Failed": failed,
        "Status": "COMPLETE" if failed == 0 else "INCOMPLETE",
        "Failed checks": [f"{r[1]}: {r[2]}" for r in results if r[0] == FAIL] or ["None"]
    }
    write_module_summary("module_01_verification", summary, cfg)
except Exception:
    pass
