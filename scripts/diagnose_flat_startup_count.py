"""
diagnose_flat_startup_count.py — Find the source of duplicated/flat values
=============================================================================
Run this to trace exactly where India's (or any country's) startup_count
became identical across multiple years.

Usage:
    python scripts\\diagnose_flat_startup_count.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

print("=" * 70)
print("  DIAGNOSING FLAT startup_count VALUES")
print("=" * 70)

COUNTRY = "India"

# ── Step 1: Check the RAW startup ecosystem data ──────────────────────────────
print(f"\n[1] Checking RAW data for {COUNTRY}...")
raw_candidates = [
    ROOT / "data" / "raw" / "startup_ecosystem_raw.csv",
    ROOT / "data" / "startup_ecosystem_raw.csv",
]
raw_path = next((p for p in raw_candidates if p.exists()), None)
if raw_path:
    raw_df = pd.read_csv(raw_path)
    country_col = next((c for c in ["country", "country_name"] if c in raw_df.columns),
                       raw_df.columns[0])
    sub = raw_df[raw_df[country_col].astype(str).str.lower() == COUNTRY.lower()]
    if "startup_count" in sub.columns and "year" in sub.columns:
        yearly = sub.groupby("year")["startup_count"].sum().sort_index()
        print(yearly.to_string())
        if yearly.tail(4).nunique() == 1:
            print(f"  >>> BUG CONFIRMED IN RAW DATA — fix Module 2 (data collection)")
        else:
            print(f"  >>> Raw data looks fine — bug is introduced LATER (Module 4/5/6)")
    else:
        print(f"  Columns found: {list(sub.columns)}")
else:
    print("  Raw file not found, skipping.")

# ── Step 2: Check INTERIM / cleaned data ──────────────────────────────────────
print(f"\n[2] Checking CLEANED/INTERIM data for {COUNTRY}...")
clean_candidates = [
    ROOT / "data" / "interim" / "master_raw.csv",
    ROOT / "data" / "processed" / "master_clean.csv",
    ROOT / "data" / "master_clean.csv",
    ROOT / "data" / "master_data.csv",
]
for path in clean_candidates:
    if path.exists():
        df = pd.read_csv(path)
        country_col = next((c for c in ["country", "country_name"] if c in df.columns),
                           df.columns[0] if len(df.columns) else None)
        if country_col and "startup_count" in df.columns:
            sub = df[df[country_col].astype(str).str.lower() == COUNTRY.lower()]
            if "year" in sub.columns:
                yearly = sub.groupby("year")["startup_count"].sum().sort_index()
                print(f"  File: {path.name}")
                print(yearly.to_string())
                if len(yearly) >= 4 and yearly.tail(4).nunique() == 1:
                    print(f"  >>> BUG PRESENT in {path.name}")
                else:
                    print(f"  >>> {path.name} looks fine")
                print()

# ── Step 3: Check FINAL features file ─────────────────────────────────────────
print(f"\n[3] Checking FINAL master_features.csv for {COUNTRY}...")
feat_candidates = [
    ROOT / "data" / "processed" / "master_features.csv",
    ROOT / "data" / "master_features.csv",
]
feat_path = next((p for p in feat_candidates if p.exists()), None)
if feat_path:
    fdf = pd.read_csv(feat_path)
    country_col = next((c for c in ["country", "country_name"] if c in fdf.columns),
                       fdf.columns[0])
    sub = fdf[fdf[country_col].astype(str).str.lower() == COUNTRY.lower()].sort_values("year")
    print(sub[["year", "startup_count"]].to_string(index=False))

    # Check if ANY row in the raw row-level data (before groupby) has dupes
    print(f"\n[3b] Row count per year for {COUNTRY} (checking for multi-row-per-year):")
    print(sub.groupby("year").size().to_string())
    if (sub.groupby("year").size() > 1).any():
        print("  >>> MULTIPLE ROWS PER YEAR DETECTED.")
        print("  >>> If Module 6 used .mean()/.first() instead of proper aggregation,")
        print("  >>> or a merge created row duplication, this would explain flat values")
        print("  >>> if the duplicated rows all carry the SAME startup_count by mistake.")
else:
    print("  master_features.csv not found.")

# ── Step 4: Check for forward-fill or interpolate calls in known scripts ──────
print(f"\n[4] Searching your scripts for risky fill operations...")
script_dir = ROOT / "scripts"
risky_patterns = ["ffill", "fillna(method", "interpolate", ".fillna(df", "bfill"]
if script_dir.exists():
    for script in sorted(script_dir.glob("*.py")):
        try:
            content = script.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = [p for p in risky_patterns if p in content]
        if hits:
            print(f"  {script.name}: contains {hits}")

print("\n" + "=" * 70)
print("  DIAGNOSIS COMPLETE")
print("=" * 70)
print("""
Read the output above top to bottom:
  - If [1] RAW data is already flat for the last years -> bug is in
    Module 2 (data collection/generation), not feature engineering.
  - If [1] is fine but [2] CLEANED data is flat -> bug is in your
    cleaning/merge step (Module 4).
  - If [2] is fine but [3] FINAL features file is flat -> bug is in
    your Module 6 feature engineering script specifically.
  - If [3b] shows multiple rows per year -> the aggregation step
    (groupby) in Module 6 is likely using the wrong function or a
    merge duplicated rows.

Share this script's full output and I'll pinpoint the exact line
to fix in your Module 6 script.
""")
