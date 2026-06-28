"""
module3_validation.py — Data Validation
=========================================

OBJECTIVE
---------
Systematically audit master_raw.csv before any modelling touches it.
Every flaw found here is cheaper to fix than after a model is trained.

CHECKS PERFORMED
----------------
1. Schema         — expected columns present, dtypes correct
2. Missing values — per column % and panel completeness (country × year)
3. Duplicates     — exact-row and key-level (country, year)
4. Range validity — domain-sensible bounds per column
5. Outliers       — IQR method per numeric column
6. Consistency    — pandemic_period flag matches year ≥ 2020

MATHEMATICAL NOTE
-----------------
Outlier boundary (IQR method):
  Q1 = 25th percentile,  Q3 = 75th percentile
  IQR = Q3 − Q1
  lower = Q1 − 1.5 × IQR
  upper = Q3 + 1.5 × IQR
Any value outside [lower, upper] is flagged.

OUTPUTS
-------
- outputs/figures/module3_*.png  (6 figures)
- outputs/reports/module3_validation_report.txt
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils import load_config, setup_logging, save_figure, write_module_summary, load_dataframe

# ── Expected schema ───────────────────────────────────────────────────────────
EXPECTED_COLS = {
    "country":                    "object",
    "year":                       "int64",
    "gdp_usd":                    "float64",
    "gdp_per_capita_usd":         "float64",
    "gdp_growth_rate":            "float64",
    "internet_penetration_pct":   "float64",
    "unemployment_rate":          "float64",
    "population":                 "float64",
    "startup_count":              "float64",
    "total_funding_usd":          "float64",
    "venture_capital_usd":        "float64",
    "startup_count_growth_rate":  "float64",
    "pandemic_period":            "int64",
}

# ── Valid value ranges ────────────────────────────────────────────────────────
VALID_RANGES = {
    "gdp_usd":                   (1e9,   1e14),
    "gdp_per_capita_usd":        (500,   200000),
    "gdp_growth_rate":           (-30,   30),       # signed — recessions are valid
    "internet_penetration_pct":  (0,     100),
    "unemployment_rate":         (0,     60),
    "startup_count_growth_rate": (-100,  300),       # signed
    "pandemic_period":           (0,     1),
}

findings = {}   # collects all check results for the report


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_schema(df, logger):
    issues = []
    missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        logger.warning(f"Schema: missing columns {missing_cols}")
    else:
        logger.info("Schema: all expected columns present")

    # Coerce year to int if loaded as float
    if "year" in df.columns and df["year"].dtype == "float64":
        df["year"] = df["year"].astype(int)

    # Whitespace in string columns
    for col in df.select_dtypes("object").columns:
        bad = df[col].dropna().str.contains(r"^\s+|\s+$").sum()
        if bad:
            issues.append(f"Whitespace in '{col}': {bad} rows")

    findings["schema_issues"] = issues if issues else ["None"]
    logger.info(f"Schema check complete: {len(issues)} issues")
    return df


def check_missing(df, logger):
    miss = (df.isnull().mean() * 100).round(2).sort_values(ascending=False)
    critical = miss[miss > 50]
    findings["missing_pct"] = miss.to_dict()
    findings["critical_missing"] = critical.to_dict() if len(critical) else {"None": 0}
    logger.info(f"Missing check: max missing = {miss.max():.1f}% ({miss.idxmax()})")

    # Panel completeness: every country should have every year
    expected_pairs = set()
    for c in df["country"].unique():
        for y in range(df["year"].min(), df["year"].max() + 1):
            expected_pairs.add((c, y))
    actual_pairs = set(zip(df["country"], df["year"]))
    gap_pairs = expected_pairs - actual_pairs
    findings["panel_gaps"] = len(gap_pairs)
    if gap_pairs:
        logger.warning(f"Panel gaps: {len(gap_pairs)} missing (country,year) combinations")
    else:
        logger.info("Panel completeness: no gaps")
    return miss


def check_duplicates(df, logger):
    exact_dups = df.duplicated().sum()
    key_dups   = df.duplicated(subset=["country", "year"]).sum()
    findings["exact_duplicates"] = int(exact_dups)
    findings["key_duplicates"]   = int(key_dups)
    logger.info(f"Duplicates: exact={exact_dups}, key-level={key_dups}")


def check_ranges(df, logger):
    range_issues = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out = df[(df[col].notna()) & ((df[col] < lo) | (df[col] > hi))]
        if len(out):
            range_issues[col] = int(len(out))
            logger.warning(f"Range violation '{col}': {len(out)} rows outside [{lo}, {hi}]")
    findings["range_issues"] = range_issues if range_issues else {"None": 0}
    logger.info(f"Range check: {len(range_issues)} columns with violations")


def check_outliers(df, logger):
    numeric = df.select_dtypes(include="number").columns.tolist()
    outlier_counts = {}
    for col in numeric:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out:
            outlier_counts[col] = int(n_out)
    findings["outlier_counts"] = outlier_counts
    logger.info(f"Outlier check: {len(outlier_counts)} columns have outliers")
    return outlier_counts


def check_consistency(df, logger):
    # pandemic_period must equal (year >= 2020)
    expected_flag = (df["year"] >= 2020).astype(int)
    mismatch = (df["pandemic_period"] != expected_flag).sum()
    findings["pandemic_flag_mismatch"] = int(mismatch)
    logger.info(f"Consistency: pandemic_period mismatches = {mismatch}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_missing_heatmap(df, config, logger):
    """Figure 1: Heatmap of missing values per column."""
    pivot = df.isnull().astype(int)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot.T, cbar=False, cmap="YlOrRd", ax=ax, yticklabels=True)
    ax.set_title("Missing Value Map (yellow = missing)", fontsize=13)
    ax.set_xlabel("Row index"); ax.set_ylabel("Column")
    plt.tight_layout()
    p = save_figure(fig, "module3_01_missing_heatmap.png", config)
    logger.info(f"Fig 1 saved: {p}")


def fig_missing_bar(df, config, logger):
    """Figure 2: Bar chart of % missing per column."""
    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        logger.info("Fig 2 skipped: no missing values")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    miss.plot(kind="bar", color="coral", ax=ax)
    ax.axhline(10, color="red", linestyle="--", linewidth=1, label="10% threshold")
    ax.set_title("Missing Values (%) by Column", fontsize=13)
    ax.set_ylabel("% Missing"); ax.set_xlabel("")
    ax.legend(); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    p = save_figure(fig, "module3_02_missing_bar.png", config)
    logger.info(f"Fig 2 saved: {p}")


def fig_outlier_boxplots(df, config, logger):
    """Figure 3: Box plots for key numeric columns."""
    cols = ["gdp_growth_rate", "internet_penetration_pct",
            "unemployment_rate", "startup_count_growth_rate"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    for ax, col in zip(axes, cols):
        df.boxplot(column=col, ax=ax, notch=False)
        ax.set_title(col.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("")
    fig.suptitle("Outlier Detection — Box Plots (IQR Method)", fontsize=13)
    plt.tight_layout()
    p = save_figure(fig, "module3_03_outlier_boxplots.png", config)
    logger.info(f"Fig 3 saved: {p}")


def fig_outlier_counts(outlier_counts, config, logger):
    """Figure 4: Bar chart of outlier counts per column."""
    if not outlier_counts:
        logger.info("Fig 4 skipped: no outliers")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = list(outlier_counts.keys())
    vals = list(outlier_counts.values())
    ax.bar(cols, vals, color="steelblue")
    ax.set_title("Number of Outliers per Column (IQR Method)", fontsize=13)
    ax.set_ylabel("Count"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    p = save_figure(fig, "module3_04_outlier_counts.png", config)
    logger.info(f"Fig 4 saved: {p}")


def fig_panel_completeness(df, config, logger):
    """Figure 5: Heatmap of panel completeness — country × year."""
    pivot = df.pivot_table(index="country", columns="year",
                           values="startup_count", aggfunc="count")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Greens",
                cbar=False, linewidths=0.5, ax=ax)
    ax.set_title("Panel Completeness — Startup Count Rows per Country × Year", fontsize=12)
    plt.tight_layout()
    p = save_figure(fig, "module3_05_panel_completeness.png", config)
    logger.info(f"Fig 5 saved: {p}")


def fig_distribution_grid(df, config, logger):
    """Figure 6: Distribution histograms for 6 key columns."""
    cols = ["gdp_growth_rate", "internet_penetration_pct", "unemployment_rate",
            "startup_count", "total_funding_usd", "startup_count_growth_rate"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flat, cols):
        data = df[col].dropna()
        ax.hist(data, bins=20, color="teal", edgecolor="white", alpha=0.8)
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.set_ylabel("Frequency")
    fig.suptitle("Distribution of Key Variables", fontsize=13)
    plt.tight_layout()
    p = save_figure(fig, "module3_06_distributions.png", config)
    logger.info(f"Fig 6 saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(config=None):
    if config is None:
        config = load_config()
    logger = setup_logging("module3_validation", config)

    logger.info("=" * 55)
    logger.info("MODULE 3 — DATA VALIDATION")
    logger.info("=" * 55)

    df = load_dataframe("master_raw.csv", stage="interim", config=config)
    logger.info(f"Loaded master_raw.csv: {df.shape}")

    # Run all checks
    df      = check_schema(df, logger)
    miss    = check_missing(df, logger)
    check_duplicates(df, logger)
    check_ranges(df, logger)
    outlier_counts = check_outliers(df, logger)
    check_consistency(df, logger)

    # Generate figures
    fig_missing_heatmap(df, config, logger)
    fig_missing_bar(df, config, logger)
    fig_outlier_boxplots(df, config, logger)
    fig_outlier_counts(outlier_counts, config, logger)
    fig_panel_completeness(df, config, logger)
    fig_distribution_grid(df, config, logger)

    # Overall health score
    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    health = round(100 * (1 - missing_cells / total_cells), 1)

    summary = {
        "Dataset shape":          str(df.shape),
        "Total cells":            total_cells,
        "Missing cells":          int(missing_cells),
        "Data health score (%)":  health,
        "Schema issues":          findings["schema_issues"],
        "Exact duplicates":       findings["exact_duplicates"],
        "Key duplicates":         findings["key_duplicates"],
        "Panel gaps":             findings["panel_gaps"],
        "Range violations":       findings["range_issues"],
        "Columns with outliers":  len(outlier_counts),
        "Outlier details":        outlier_counts,
        "Pandemic flag mismatch": findings["pandemic_flag_mismatch"],
        "Status":                 "PASS" if health >= 80 else "NEEDS CLEANING",
    }

    write_module_summary("module3_validation", summary, config)

    print("\n" + "="*55)
    print("  MODULE 3 — VALIDATION REPORT")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ 6 figures saved to outputs/figures/")
    print(f"  ✓ Report saved to outputs/reports/")
    print("="*55)

    return df, findings


if __name__ == "__main__":
    run()
