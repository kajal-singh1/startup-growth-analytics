"""
module4_cleaning.py — Data Cleaning
=====================================

OBJECTIVE
---------
Produce a fully clean dataset ready for feature engineering and modelling.
Addresses every issue surfaced in Module 3 Validation.

STEPS
-----
1. Drop columns with > 60% missing (none expected, but handled)
2. Impute missing values
   - Numeric: forward-fill within each country (time series logic),
     then backward-fill, then global median as last resort
   - Why per-country fill? GDP of India in 2016 is best estimated
     from India's 2015 value, not the global mean across all countries.
3. Handle outliers — Winsorize at 1st/99th percentile
   - We winsorize (clip) rather than drop because removing entire rows
     would create panel gaps that break time-series analysis.
4. Scale features — StandardScaler
   - Zero mean, unit variance:  z = (x − μ) / σ
   - Stored alongside raw values so both are available downstream.
5. Store clean dataset to data/processed/ and SQLite master_dataset table.

MATHEMATICAL NOTE
-----------------
Winsorizing at percentiles p_low and p_high:
  x_clean = clip(x, quantile(x, p_low), quantile(x, p_high))

Forward fill (within group g):
  x̂(t) = x(t) if x(t) is not NaN, else x̂(t−1)

INPUTS
------
- data/interim/master_raw.csv

OUTPUTS
-------
- data/processed/master_clean.csv
- SQLite table: master_dataset (populated)
- outputs/figures/module4_*.png  (5 figures)
- outputs/reports/module4_cleaning_report.txt
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from utils import (load_config, setup_logging, save_figure,
                   save_dataframe, load_dataframe,
                   get_db_connection, write_module_summary, set_seeds)

NUMERIC_COLS = [
    "gdp_usd", "gdp_per_capita_usd", "gdp_growth_rate",
    "internet_penetration_pct", "unemployment_rate", "population",
    "startup_count", "total_funding_usd", "venture_capital_usd",
    "startup_count_growth_rate",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING STEPS
# ─────────────────────────────────────────────────────────────────────────────

def drop_high_missing(df, threshold=0.6, logger=None):
    """Drop columns missing > threshold fraction of values."""
    miss_frac = df.isnull().mean()
    drop_cols = miss_frac[miss_frac > threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
        if logger:
            logger.warning(f"Dropped high-missing columns: {drop_cols}")
    else:
        if logger:
            logger.info("No columns dropped (none exceeded 60% missing)")
    return df


def impute_missing(df, logger):
    """
    Impute numeric columns: forward-fill → backward-fill → global median.
    Category/string columns: mode fill.
    """
    df = df.sort_values(["country", "year"]).copy()
    before = df.isnull().sum().sum()

    # Per-country time-ordered fill
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_present] = (
        df.groupby("country")[numeric_present]
        .transform(lambda g: g.ffill().bfill())
    )

    # Remaining gaps: global median
    for col in numeric_present:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.info(f"  Median-filled residual NaN in '{col}' with {median:.4g}")

    after = df.isnull().sum().sum()
    logger.info(f"Imputation: {before} → {after} missing cells")
    return df


def winsorize(df, lower_pct=0.01, upper_pct=0.99, logger=None):
    """
    Clip extreme outliers at 1st and 99th percentiles.

    Why not drop?  Dropping rows creates panel gaps that break time-series
    continuity. Clipping keeps all 135 rows intact.
    """
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    winsor_log = {}
    for col in numeric_present:
        lo = df[col].quantile(lower_pct)
        hi = df[col].quantile(upper_pct)
        before_out = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        if before_out:
            winsor_log[col] = int(before_out)
    if logger:
        logger.info(f"Winsorized: {winsor_log}")
    return df, winsor_log


def scale_features(df, logger):
    """
    Add z-score scaled columns (suffix _scaled) for ML modules.
    Raw columns are kept untouched.
    """
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[numeric_present])
    scaled_cols = [c + "_scaled" for c in numeric_present]
    scaled_df = pd.DataFrame(scaled_vals, columns=scaled_cols, index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    logger.info(f"Scaled {len(numeric_present)} columns → added {len(scaled_cols)} *_scaled columns")
    return df, scaler


def store_to_db(df, config, logger):
    """Write clean dataset to master_dataset table."""
    conn = get_db_connection(config)
    conn.execute("DELETE FROM master_dataset")

    # Rename to match DB schema
    if "total_funding_usd" in df.columns:
        df = df.rename(columns={"total_funding_usd": "startup_funding_usd"})

    db_cols = [
        "country", "year", "startup_count", "startup_funding_usd",
        "venture_capital_usd", "startup_count_growth_rate",
        "gdp_usd", "gdp_per_capita_usd", "gdp_growth_rate",
        "fdi_pct_gdp", "internet_penetration_pct",
        "research_expenditure_pct", "unemployment_rate",
        "tertiary_enrollment_pct", "population", "pandemic_period",
    ]
    # Only insert columns that exist in df
    insert_cols = [c for c in db_cols if c in df.columns]
    placeholders = ", ".join(["?"] * len(insert_cols))
    col_names    = ", ".join(insert_cols)

    inserted = 0
    for _, row in df.iterrows():
        vals = tuple(
            None if pd.isna(row[c]) else
            int(row[c]) if c in ("year", "pandemic_period") else
            float(row[c]) if c != "country" else str(row[c])
            for c in insert_cols
        )
        conn.execute(
            f"INSERT INTO master_dataset ({col_names}) VALUES ({placeholders})", vals
        )
        inserted += 1

    conn.commit()
    conn.close()
    logger.info(f"Stored {inserted} rows to master_dataset table")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_missing_before_after(raw, clean, config, logger):
    """Figure 1: Missing values before vs after cleaning."""
    before = raw.isnull().sum()
    after  = clean[[c for c in raw.columns if c in clean.columns]].isnull().sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    before[before > 0].plot(kind="bar", ax=axes[0], color="coral")
    axes[0].set_title("Missing Values BEFORE Cleaning"); axes[0].set_ylabel("Count")
    after_nonzero = after[after > 0]
    if len(after_nonzero):
        after_nonzero.plot(kind="bar", ax=axes[1], color="steelblue")
    else:
        axes[1].text(0.5, 0.5, "Zero missing values\nafter cleaning",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=14)
    axes[1].set_title("Missing Values AFTER Cleaning")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    save_figure(fig, "module4_01_missing_before_after.png", config)
    logger.info("Fig 1 saved")


def fig_outlier_before_after(raw, clean, config, logger):
    """Figure 2: Box plots before/after winsorizing for growth rates."""
    cols = [c for c in ["gdp_growth_rate", "startup_count_growth_rate", "unemployment_rate"]
            if c in raw.columns]
    fig, axes = plt.subplots(2, len(cols), figsize=(14, 7))
    for i, col in enumerate(cols):
        raw[col].dropna().plot(kind="box", ax=axes[0, i], title=f"BEFORE\n{col}")
        clean[col].dropna().plot(kind="box", ax=axes[1, i], title=f"AFTER\n{col}")
    fig.suptitle("Outlier Treatment — Winsorizing at 1st/99th Percentile", fontsize=13)
    plt.tight_layout()
    save_figure(fig, "module4_02_outlier_before_after.png", config)
    logger.info("Fig 2 saved")


def fig_scaled_distributions(clean, config, logger):
    """Figure 3: Distributions of scaled features."""
    scaled_cols = [c for c in clean.columns if c.endswith("_scaled")][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flat, scaled_cols):
        ax.hist(clean[col].dropna(), bins=20, color="teal", edgecolor="white", alpha=0.8)
        ax.set_title(col.replace("_scaled", "").replace("_", " "), fontsize=9)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
    fig.suptitle("Scaled Feature Distributions (should be ~N(0,1))", fontsize=13)
    plt.tight_layout()
    save_figure(fig, "module4_03_scaled_distributions.png", config)
    logger.info("Fig 3 saved")


def fig_correlation_heatmap(clean, config, logger):
    """Figure 4: Correlation heatmap of key raw features."""
    cols = [c for c in NUMERIC_COLS if c in clean.columns]
    corr = clean[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, square=True,
                linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap — Clean Features", fontsize=13)
    plt.tight_layout()
    save_figure(fig, "module4_04_correlation_heatmap.png", config)
    logger.info("Fig 4 saved")


def fig_target_by_country(clean, config, logger):
    """Figure 5: Target variable (startup_count_growth_rate) by country."""
    if "startup_count_growth_rate" not in clean.columns:
        return
    grouped = (clean.groupby("country")["startup_count_growth_rate"]
               .mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in grouped.values]
    grouped.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Mean Startup Count Growth Rate by Country (Target Variable)", fontsize=13)
    ax.set_ylabel("Mean Growth Rate (%)"); ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    save_figure(fig, "module4_05_target_by_country.png", config)
    logger.info("Fig 5 saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(config=None):
    if config is None:
        config = load_config()
    set_seeds(config)
    logger = setup_logging("module4_cleaning", config)

    logger.info("=" * 55)
    logger.info("MODULE 4 — DATA CLEANING")
    logger.info("=" * 55)

    raw = load_dataframe("master_raw.csv", stage="interim", config=config)
    logger.info(f"Loaded master_raw.csv: {raw.shape}")
    raw_snapshot = raw.copy()

    # Apply cleaning steps
    df = drop_high_missing(raw.copy(), threshold=0.6, logger=logger)
    df = impute_missing(df, logger)
    df, winsor_log = winsorize(df, logger=logger)
    df, scaler     = scale_features(df, logger)

    # Verify clean
    remaining_miss = df.isnull().sum().sum()
    logger.info(f"Remaining missing after cleaning: {remaining_miss}")

    # Save
    csv_path = save_dataframe(df, "master_clean.csv", stage="processed", config=config)
    logger.info(f"Saved: {csv_path}")
    inserted = store_to_db(df, config, logger)

    # Figures
    fig_missing_before_after(raw_snapshot, df, config, logger)
    fig_outlier_before_after(raw_snapshot, df, config, logger)
    fig_scaled_distributions(df, config, logger)
    fig_correlation_heatmap(df, config, logger)
    fig_target_by_country(df, config, logger)

    summary = {
        "Raw shape":              str(raw_snapshot.shape),
        "Clean shape":            str(df.shape),
        "Missing before":         int(raw_snapshot.isnull().sum().sum()),
        "Missing after":          int(remaining_miss),
        "Winsorized columns":     winsor_log,
        "Scaled columns added":   len([c for c in df.columns if c.endswith("_scaled")]),
        "DB rows inserted":       inserted,
        "Output CSV":             str(csv_path),
        "Status":                 "COMPLETE",
    }
    write_module_summary("module4_cleaning", summary, config)

    print("\n" + "="*55)
    print("  MODULE 4 — CLEANING REPORT")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ 5 figures saved to outputs/figures/")
    print(f"  ✓ master_clean.csv → data/processed/")
    print(f"  ✓ master_dataset table populated in SQLite")
    print("="*55)

    return df


if __name__ == "__main__":
    run()
