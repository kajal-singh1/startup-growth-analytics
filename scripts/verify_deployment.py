"""
verify_deployment.py — Module 13: Pre-Deployment Checklist
=============================================================
Run this BEFORE pushing to GitHub / deploying to Streamlit Cloud
or Render. Catches the most common deployment failures early.

Run:
    python scripts\\verify_deployment.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
results = []


def check(label, condition, detail="", warn_only=False):
    if condition:
        status = PASS
    elif warn_only:
        status = WARN
    else:
        status = FAIL
    results.append((status, label, detail))
    print(f"{status}  {label}" + (f" — {detail}" if detail else ""))


print("\n" + "=" * 60)
print("  MODULE 13 — PRE-DEPLOYMENT VERIFICATION")
print("=" * 60 + "\n")

# ── 1. Required files exist ───────────────────────────────────────────────────
print("--- Required Files ---")
required_files = [
    "scripts/dashboard.py",
    "scripts/api.py",
    "requirements_dashboard.txt",
    "requirements_api.txt",
    "render.yaml",
    "Procfile",
    ".streamlit/config.toml",
]
for f in required_files:
    check(f"File: {f}", (ROOT / f).exists())

# ── 2. Data files present (needed for dashboard to render) ────────────────────
print("\n--- Data Files ---")
data_candidates = [
    "data/processed/master_features.csv",
    "data/master_features.csv",
]
data_found = any((ROOT / f).exists() for f in data_candidates)
check("master_features.csv present", data_found,
      "Dashboard will show empty pages without this")

forecast_found = (ROOT / "data" / "forecasts_2027.csv").exists() or \
                  (ROOT / "data" / "processed" / "forecasts_2027.csv").exists()
check("forecasts_2027.csv present", forecast_found, warn_only=True)

cluster_found = (ROOT / "data" / "cluster_assignments.csv").exists() or \
                 (ROOT / "data" / "processed" / "cluster_assignments.csv").exists()
check("cluster_assignments.csv present", cluster_found, warn_only=True)

# ── 3. Model file for API predictions ─────────────────────────────────────────
print("\n--- Model Files ---")
model_candidates = [
    "models/best_model.pkl",
    "models/saved/best_model.pkl",
]
model_found = any((ROOT / f).exists() for f in model_candidates)
check("best_model.pkl present", model_found,
      "/predict endpoint will use heuristic fallback without this",
      warn_only=True)

# ── 4. Figures exist for dashboard galleries ──────────────────────────────────
print("\n--- Figure Folders ---")
fig_base = ROOT / "data" / "outputs" / "figures"
for mod in [3, 5, 6, 7, 8, 9, 10]:
    d = fig_base / f"module{mod}"
    n = len(list(d.glob("*.png"))) if d.exists() else 0
    check(f"module{mod} figures", n > 0, f"{n} PNGs found", warn_only=True)

# ── 5. .gitignore doesn't exclude what we need ────────────────────────────────
print("\n--- .gitignore Safety Check ---")
gitignore_path = ROOT / ".gitignore"
if gitignore_path.exists():
    gi_content = gitignore_path.read_text()
    # Only check ACTIVE rules (strip comments and blank lines)
    active_lines = [
        line.strip() for line in gi_content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    dangerous_patterns = ["data/", "*.csv", "models/", "data/processed/"]
    found_dangerous = [p for p in dangerous_patterns if p in active_lines]
    check(".gitignore doesn't block deployment data",
          len(found_dangerous) == 0,
          f"Found risky patterns: {found_dangerous}" if found_dangerous else "")
else:
    check(".gitignore exists", False, warn_only=True)

# ── 6. Requirements files are lean (no TensorFlow/heavy deps for dashboard) ───
print("\n--- Requirements Sanity ---")
dash_req = (ROOT / "requirements_dashboard.txt").read_text() \
    if (ROOT / "requirements_dashboard.txt").exists() else ""
check("Dashboard requirements exclude TensorFlow",
      "tensorflow" not in dash_req.lower(),
      "Streamlit Cloud free tier has 1GB RAM — TensorFlow will OOM")
check("Dashboard requirements exclude XGBoost/LightGBM",
      "xgboost" not in dash_req.lower() and "lightgbm" not in dash_req.lower(),
      "Dashboard only reads CSVs, doesn't need ML libs")

# ── 7. File sizes (GitHub has 100MB hard limit, 50MB warning) ─────────────────
print("\n--- File Size Check ---")
large_files = []
for f in ROOT.rglob("*"):
    if f.is_file() and ".git" not in str(f) and "venv" not in str(f):
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            large_files.append((str(f.relative_to(ROOT)), round(size_mb, 1)))
check("No files exceed 50MB", len(large_files) == 0,
      str(large_files) if large_files else "")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for r in results if r[0] == PASS)
warned = sum(1 for r in results if r[0] == WARN)
failed = sum(1 for r in results if r[0] == FAIL)

print("\n" + "=" * 60)
print(f"  RESULTS: {passed} passed, {warned} warnings, {failed} failed (of {total})")
print("=" * 60)

if failed == 0:
    print("\n  Ready to deploy! Follow DEPLOYMENT_GUIDE.md next.")
else:
    print(f"\n  Fix the {failed} FAILED item(s) above before deploying.")
    print("  WARN items are optional but recommended.")

sys.exit(1 if failed > 0 else 0)
