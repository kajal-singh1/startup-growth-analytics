"""
audit_project.py — Full End-to-End Project Audit
===================================================
Runs every check needed to confirm the project is deployment-ready:
  1. Syntax validity of every script
  2. The exact two bug patterns already found and fixed (Module 6, Module 9)
     re-checked across ALL scripts, not just those two
  3. Data file consistency — do all modules read from the same final dataset?
  4. Column name consistency — does 'country' vs 'country_name' break anything?
  5. Output file presence — does every module's expected output actually exist?
  6. Cross-module data flow — does Module N's input match Module N-1's output?
  7. Deployment readiness (re-runs the Module 13 check)

Run:
    python scripts\\audit_project.py

This does NOT modify any files. It only reports.
"""

import ast
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS, FAIL, WARN = "  [PASS]", "  [FAIL]", "  [WARN]"
results = []


def check(label, condition, detail="", warn_only=False):
    status = PASS if condition else (WARN if warn_only else FAIL)
    results.append((status, label, detail))
    print(f"{status}  {label}" + (f" — {detail}" if detail else ""))


print("\n" + "=" * 70)
print("  FULL PROJECT AUDIT")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTAX CHECK — every .py file in scripts/ and src/
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 1. Syntax Validity ---")
py_files = list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py"))
py_files = [f for f in py_files if "__pycache__" not in str(f) and "venv" not in str(f)]

for f in py_files:
    try:
        ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        check(f"Syntax: {f.relative_to(ROOT)}", True)
    except SyntaxError as e:
        check(f"Syntax: {f.relative_to(ROOT)}", False, f"Line {e.lineno}: {e.msg}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUG PATTERN RE-CHECK — across ALL scripts, not just the 2 already fixed
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 2. Known Bug Pattern Re-Check ---")

for f in py_files:
    content = f.read_text(encoding="utf-8", errors="ignore")
    rel = f.relative_to(ROOT)

    # Pattern A: global quantile/clip without groupby nearby (Module 6 bug)
    quantile_lines = [i+1 for i, line in enumerate(content.splitlines())
                      if ".quantile(" in line or "np.clip(" in line]
    for ln in quantile_lines:
        # Check if groupby appears within 10 lines before it
        lines = content.splitlines()
        window = "\n".join(lines[max(0, ln-10):ln])
        has_groupby_context = "groupby" in window
        check(f"Quantile/clip at {rel}:{ln} is grouped",
              has_groupby_context,
              "No nearby groupby() — verify this isn't a repeat of the Module 6 bug",
              warn_only=True)

    # Pattern B: hardcoded "country" without fallback (Module 9 bug)
    if '"country"' in content or "'country'" in content:
        has_fallback = ("country_name" in content) or ("country_code" in content)
        check(f'"country" column usage in {rel} has fallback',
              has_fallback,
              "Hardcodes 'country' without checking 'country_name'/'country_code' — "
              "verify this matches your actual CSV schema",
              warn_only=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA FILE CONSISTENCY — what does each module actually read/write?
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3. Data File Read/Write Map ---")

io_pattern = re.compile(
    r'(read_csv|to_csv|load_dataframe|save_dataframe)\([^)]*?[\'"]([^\'\"]+\.csv)[\'"]'
)
file_map = {}
for f in py_files:
    content = f.read_text(encoding="utf-8", errors="ignore")
    matches = io_pattern.findall(content)
    if matches:
        file_map[str(f.relative_to(ROOT))] = matches

for script, ops in file_map.items():
    for op, fname in ops:
        direction = "WRITES" if op in ("to_csv", "save_dataframe") else "READS"
        print(f"    {script:45s} {direction:6s} {fname}")

# Check: does every module that READS master_features.csv have a corresponding
# WRITER somewhere in the project?
writers = {fname for ops in file_map.values() for op, fname in ops
           if op in ("to_csv", "save_dataframe") and "master_features" in fname}
readers = {(script, fname) for script, ops in file_map.items() for op, fname in ops
          if op in ("read_csv", "load_dataframe") and "master_features" in fname}

check("master_features.csv has a writer script", len(writers) > 0,
      f"Writers found: {writers}" if writers else "NO WRITER FOUND — readers will fail")

# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPECTED OUTPUT FILES — do they actually exist on disk?
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 4. Expected Output Files Present ---")

expected_outputs = [
    ("data/processed/master_features.csv", "Module 6 output"),
    ("data/processed/master_dataset.csv",   "Module 5/upstream output"),
    ("data/cluster_assignments.csv",        "Module 9 output"),
    ("data/forecasts_2027.csv",             "Module 10 output"),
    ("data/processed/did_results.csv",      "Module 7 output"),
    ("models/best_model.pkl",               "Module 8 model (optional)"),
    ("models/saved/best_model.pkl",         "Module 8 model (alt path, optional)"),
]
for relpath, desc in expected_outputs:
    full = ROOT / relpath
    is_optional = "optional" in desc
    check(f"{relpath}", full.exists(), desc, warn_only=is_optional)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SCHEMA CONSISTENCY — does master_features.csv have the columns every
#    downstream module expects?
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 5. master_features.csv Schema Check ---")

feat_path = ROOT / "data" / "processed" / "master_features.csv"
if feat_path.exists():
    try:
        import pandas as pd
        df = pd.read_csv(feat_path, nrows=5)
        cols = set(df.columns)

        required_by_module = {
            "country identifier": (["country", "country_name", "country_code"], cols),
            "year":               (["year"], cols),
            "startup_count":      (["startup_count"], cols),
        }
        for label, (candidates, available) in required_by_module.items():
            found = [c for c in candidates if c in available]
            check(f"Schema has {label}", len(found) > 0,
                  f"Found: {found}" if found else f"Looked for: {candidates}")

        # Re-verify the flat-value bug doesn't exist anymore for ANY country
        full_df = pd.read_csv(feat_path)
        country_col = next((c for c in ["country", "country_name"] if c in full_df.columns), None)
        if country_col and "startup_count" in full_df.columns and "year" in full_df.columns:
            flat_countries = []
            for c in full_df[country_col].unique():
                sub = full_df[full_df[country_col] == c].sort_values("year")
                if len(sub) >= 4 and sub["startup_count"].tail(4).nunique() == 1:
                    flat_countries.append(c)
            check("No countries have flat startup_count (last 4 years)",
                  len(flat_countries) == 0,
                  f"Still flat: {flat_countries}" if flat_countries else "")

        # Re-verify clustering would find real countries, not row indices
        check("country count is reasonable (not 150 fake rows)",
              df[country_col].nunique() if country_col in df.columns else 0 <= 20,
              warn_only=True) if country_col else None

    except ImportError:
        check("pandas available for schema check", False, "pip install pandas")
    except Exception as e:
        check("master_features.csv readable", False, str(e))
else:
    check("master_features.csv exists for schema check", False)

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLUSTER ASSIGNMENTS — re-verify the Module 9 fix actually took
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 6. Cluster Assignments Sanity ---")
cluster_path = ROOT / "data" / "cluster_assignments.csv"
if cluster_path.exists():
    try:
        import pandas as pd
        cdf = pd.read_csv(cluster_path)
        country_col = next((c for c in cdf.columns if "country" in c.lower()), cdf.columns[0])
        n_unique = cdf[country_col].nunique()
        has_placeholders = cdf[country_col].astype(str).str.match(r"^Country_\d+$").any()
        check("cluster_assignments.csv has real country names",
              not has_placeholders,
              f"{n_unique} unique values found" + (" — PLACEHOLDER NAMES DETECTED" if has_placeholders else ""))
        check("cluster_assignments.csv has reasonable country count",
              5 <= n_unique <= 25,
              f"{n_unique} countries (expected ~15)")
    except Exception as e:
        check("cluster_assignments.csv readable", False, str(e))
else:
    check("cluster_assignments.csv exists", False, warn_only=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. DASHBOARD-SPECIFIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 7. Dashboard Deprecation & Integration Check ---")
dash_path = ROOT / "scripts" / "dashboard.py"
if dash_path.exists():
    content = dash_path.read_text(encoding="utf-8", errors="ignore")
    check("No deprecated use_container_width", "use_container_width" not in content,
          "Run the find-replace fix again if this fails")
    check("No deprecated use_column_width", "use_column_width" not in content,
          "Run the find-replace fix again if this fails")
    check("No leftover 'MSc Data Science Portfolio Project' text",
          "MSc Data Science Portfolio Project" not in content)
    check("No leftover Module Completion Status table",
          "Module Completion Status" not in content)
else:
    check("scripts/dashboard.py exists", False)

# ─────────────────────────────────────────────────────────────────────────────
# 8. RUN MODULE 13's DEPLOYMENT CHECKS TOO
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 8. Deployment Readiness (Module 13) ---")
deploy_files = [
    "requirements_dashboard.txt", "requirements_api.txt",
    "render.yaml", "Procfile", ".streamlit/config.toml", ".gitignore",
]
for f in deploy_files:
    check(f"Deploy file: {f}", (ROOT / f).exists())

gi_path = ROOT / ".gitignore"
if gi_path.exists():
    gi_lines = [l.strip() for l in gi_path.read_text().splitlines()
               if l.strip() and not l.strip().startswith("#")]
    dangerous = [p for p in ["data/", "*.csv", "models/", "data/processed/"] if p in gi_lines]
    check(".gitignore doesn't block deployment data", len(dangerous) == 0,
          f"Found: {dangerous}" if dangerous else "")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for r in results if r[0] == PASS)
warned = sum(1 for r in results if r[0] == WARN)
failed = sum(1 for r in results if r[0] == FAIL)

print("\n" + "=" * 70)
print(f"  RESULTS: {passed} passed, {warned} warnings, {failed} failed (of {total})")
print("=" * 70)

if failed > 0:
    print("\n  FAILED ITEMS (must fix):")
    for status, label, detail in results:
        if status == FAIL:
            print(f"    - {label}: {detail}")

if warned > 0:
    print("\n  WARNINGS (review, may be intentional):")
    for status, label, detail in results:
        if status == WARN:
            print(f"    - {label}: {detail}")

print()
if failed == 0:
    print("  ✅ No blocking issues found. Review warnings above, then deploy.")
else:
    print(f"  ❌ Fix the {failed} failed item(s) above before deploying.")

sys.exit(1 if failed > 0 else 0)
