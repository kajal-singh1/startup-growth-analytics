"""
module5_features.py — Feature Engineering
==========================================

OBJECTIVE
---------
Create 7 new meaningful features from the clean dataset that give ML
models richer signal than raw indicators alone.

WHY FEATURE ENGINEERING MATTERS
---------------------------------
Raw variables (GDP in USD, startup count) have very different scales
and mix absolute size with structural quality. Derived features let us
ask better questions:
  - Is a country's startup density high relative to its population?
  - How efficiently does investment convert to new startups?
  - How digitally ready is the ecosystem as a whole?

FEATURES CREATED
----------------
1. startup_density
   = startup_count / (population / 1_000_000)
   Startups per million people — normalizes for country size.

2. funding_per_startup_mn
   = startup_funding_usd / (startup_count × 1_000_000)
   Average funding per startup in $M — investment quality signal.

3. innovation_score
   = 0.4 × internet_norm + 0.4 × gdp_pc_norm + 0.2 × rd_norm
   Composite of digital access, wealth, and R&D (min-max normalized).
   Weights from GII methodology (simplified).

4. digital_readiness_score
   = 0.6 × internet_norm + 0.4 × gdp_pc_norm
   Focus on connectivity and purchasing power.

5. investment_efficiency
   = startup_count / (startup_funding_usd / 1_000_000_000)
   Startups produced per $1B invested — efficiency of capital deployment.

6. economic_momentum
   = gdp_growth_rate − unemployment_rate
   Positive when growth is outpacing unemployment pressure.

7. pandemic_interaction
   = pandemic_period × internet_penetration_pct
   Interaction term for causal inference: captures whether digital
   access amplified or dampened pandemic effects on startup growth.

MATHEMATICAL NOTE — min-max normalization:
  x_norm = (x − x_min) / (x_max − x_min) ∈ [0, 1]

INPUTS
------
- data/processed/master_clean.csv

OUTPUTS
-------
- data/processed/master_features.csv
- SQLite table: engineered_features
- outputs/figures/module5_*.png  (5 figures)
- outputs/reports/module5_features_report.txt
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
from utils import (load_config, setup_logging, save_figure,
                   save_dataframe, load_dataframe,
                   get_db_connection, write_module_summary, set_seeds)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — min-max normalization
# ─────────────────────────────────────────────────────────────────────────────
def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CREATION
# ─────────────────────────────────────────────────────────────────────────────
def create_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    df = df.copy()

    # Restore total_funding column name if it was renamed
    if "startup_funding_usd" in df.columns and "total_funding_usd" not in df.columns:
        df["total_funding_usd"] = df["startup_funding_usd"]

    # ── 1. Startup density ────────────────────────────────────────────────────
    # Need population; it was dropped if >60% missing — reconstruct via gdp/gdp_pc
    if "population" not in df.columns and "gdp_usd" in df.columns and "gdp_per_capita_usd" in df.columns:
        df["population"] = df["gdp_usd"] / df["gdp_per_capita_usd"]
        logger.info("Reconstructed 'population' from gdp_usd / gdp_per_capita_usd")

    if "population" in df.columns:
        pop_millions = df["population"].replace(0, np.nan) / 1_000_000
        df["startup_density"] = df["startup_count"] / pop_millions
    else:
        df["startup_density"] = np.nan
        logger.warning("startup_density = NaN (population column unavailable)")

    # ── 2. Funding per startup ($M) ───────────────────────────────────────────
    denom = df["startup_count"].replace(0, np.nan)
    df["funding_per_startup_mn"] = df["total_funding_usd"] / (denom * 1_000_000)

    # ── 3. Innovation score ───────────────────────────────────────────────────
    internet_norm = minmax_norm(df["internet_penetration_pct"])
    gdp_pc_norm   = minmax_norm(df["gdp_per_capita_usd"])

    # R&D: many countries have NaN — fill with 0 (no data ≈ no reported R&D)
    rd_col = "research_expenditure_pct"
    if rd_col in df.columns:
        rd_norm = minmax_norm(df[rd_col].fillna(0))
    else:
        rd_norm = pd.Series(0, index=df.index)

    df["innovation_score"] = (0.4 * internet_norm +
                               0.4 * gdp_pc_norm +
                               0.2 * rd_norm).round(4)

    # ── 4. Digital readiness score ────────────────────────────────────────────
    df["digital_readiness_score"] = (0.6 * internet_norm +
                                      0.4 * gdp_pc_norm).round(4)

    # ── 5. Investment efficiency (startups per $1B) ───────────────────────────
    funding_bn = df["total_funding_usd"].replace(0, np.nan) / 1_000_000_000
    df["investment_efficiency"] = (df["startup_count"] / funding_bn).round(4)

    # ── 6. Economic momentum ──────────────────────────────────────────────────
    df["economic_momentum"] = (df["gdp_growth_rate"] - df["unemployment_rate"]).round(4)

    # ── 7. Pandemic interaction term ──────────────────────────────────────────
    df["pandemic_interaction"] = (
        df["pandemic_period"] * df["internet_penetration_pct"]
    ).round(4)

    new_features = [
        "startup_density", "funding_per_startup_mn", "innovation_score",
        "digital_readiness_score", "investment_efficiency",
        "economic_momentum", "pandemic_interaction",
    ]
    logger.info(f"Created {len(new_features)} new features: {new_features}")

    # Log feature stats
    for f in new_features:
        col = df[f]
        logger.info(f"  {f}: mean={col.mean():.3g}, std={col.std():.3g}, "
                    f"null={col.isnull().sum()}")

    return df, new_features


def store_features_to_db(df, new_features, config, logger):
    """Write engineered features to SQLite engineered_features table."""
    conn = get_db_connection(config)
    conn.execute("DELETE FROM engineered_features")

    # Map what we have to what the table expects
    db_cols = {
        "startup_density":        "startup_density",
        "funding_per_startup_mn": "funding_per_capita",
        "innovation_score":       "innovation_score",
        "digital_readiness_score":"digital_readiness_score",
        "investment_efficiency":  "investment_efficiency",
    }

    inserted = 0
    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO engineered_features
                (country, year, startup_density, funding_per_capita,
                 innovation_score, digital_readiness_score, investment_efficiency)
            VALUES (?,?,?,?,?,?,?)
        """, (
            str(row["country"]), int(row["year"]),
            float(row["startup_density"])     if pd.notna(row.get("startup_density")) else None,
            float(row["funding_per_startup_mn"]) if pd.notna(row.get("funding_per_startup_mn")) else None,
            float(row["innovation_score"]),
            float(row["digital_readiness_score"]),
            float(row["investment_efficiency"]) if pd.notna(row.get("investment_efficiency")) else None,
        ))
        inserted += 1

    conn.commit()
    conn.close()
    logger.info(f"Stored {inserted} rows to engineered_features table")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_innovation_score_map(df, config, logger):
    """Figure 1: Innovation score by country (latest year)."""
    latest = df[df["year"] == df["year"].max()].copy()
    latest = latest.sort_values("innovation_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(latest)))
    ax.barh(latest["country"], latest["innovation_score"], color=colors)
    ax.set_title(f"Innovation Score by Country ({df['year'].max()})", fontsize=13)
    ax.set_xlabel("Innovation Score (0–1)")
    plt.tight_layout()
    save_figure(fig, "module5_01_innovation_score.png", config)
    logger.info("Fig 1 saved")


def fig_startup_density(df, config, logger):
    """Figure 2: Startup density trend for top 6 countries."""
    if df["startup_density"].isnull().all():
        logger.info("Fig 2 skipped: startup_density all null")
        return
    top6 = (df.groupby("country")["startup_density"].mean()
              .sort_values(ascending=False).head(6).index)
    fig, ax = plt.subplots(figsize=(12, 5))
    for country in top6:
        sub = df[df["country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["startup_density"], marker="o", label=country)
    ax.set_title("Startup Density (per million people) — Top 6 Countries", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Startups per million people")
    ax.legend(fontsize=8); plt.tight_layout()
    save_figure(fig, "module5_02_startup_density.png", config)
    logger.info("Fig 2 saved")


def fig_feature_correlations(df, new_features, config, logger):
    """Figure 3: Correlation of new features vs target."""
    target = "startup_count_growth_rate"
    if target not in df.columns:
        return
    corrs = {}
    for f in new_features:
        if f in df.columns:
            corrs[f] = df[f].corr(df[target])
    corr_s = pd.Series(corrs).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_s.values]
    corr_s.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of Engineered Features with Target Variable", fontsize=13)
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    save_figure(fig, "module5_03_feature_target_correlation.png", config)
    logger.info("Fig 3 saved")


def fig_economic_momentum(df, config, logger):
    """Figure 4: Economic momentum (GDP growth - unemployment) over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = df.pivot_table(index="year", columns="country",
                           values="economic_momentum", aggfunc="mean")
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], alpha=0.6, linewidth=1.5, label=col)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axvspan(2019.5, 2020.5, alpha=0.15, color="red", label="COVID-19 shock")
    ax.set_title("Economic Momentum (GDP Growth − Unemployment Rate) by Country", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Momentum (pp)")
    ax.legend(fontsize=7, ncol=3); plt.tight_layout()
    save_figure(fig, "module5_04_economic_momentum.png", config)
    logger.info("Fig 4 saved")


def fig_investment_efficiency(df, config, logger):
    """Figure 5: Investment efficiency by country — pre vs post pandemic."""
    pre  = df[df["pandemic_period"] == 0].groupby("country")["investment_efficiency"].mean()
    post = df[df["pandemic_period"] == 1].groupby("country")["investment_efficiency"].mean()
    comp = pd.DataFrame({"Pre-pandemic": pre, "Post-pandemic": post}).dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(comp))
    w = 0.35
    ax.bar(x - w/2, comp["Pre-pandemic"],  width=w, label="Pre-pandemic",  color="#3498db")
    ax.bar(x + w/2, comp["Post-pandemic"], width=w, label="Post-pandemic", color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(comp.index, rotation=45, ha="right")
    ax.set_title("Investment Efficiency (Startups per $1B) — Pre vs Post Pandemic", fontsize=13)
    ax.set_ylabel("Startups per $1B"); ax.legend(); plt.tight_layout()
    save_figure(fig, "module5_05_investment_efficiency.png", config)
    logger.info("Fig 5 saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(config=None):
    if config is None:
        config = load_config()
    set_seeds(config)
    logger = setup_logging("module5_features", config)

    logger.info("=" * 55)
    logger.info("MODULE 5 — FEATURE ENGINEERING")
    logger.info("=" * 55)

    df = load_dataframe("master_clean.csv", stage="processed", config=config)
    logger.info(f"Loaded master_clean.csv: {df.shape}")

    df, new_features = create_features(df, logger)

    csv_path = save_dataframe(df, "master_features.csv", stage="processed", config=config)
    logger.info(f"Saved: {csv_path}")

    inserted = store_features_to_db(df, new_features, config, logger)

    fig_innovation_score_map(df, config, logger)
    fig_startup_density(df, config, logger)
    fig_feature_correlations(df, new_features, config, logger)
    fig_economic_momentum(df, config, logger)
    fig_investment_efficiency(df, config, logger)

    null_counts = {f: int(df[f].isnull().sum()) for f in new_features if f in df.columns}
    summary = {
        "Input shape":           str(df.shape),
        "New features":          new_features,
        "Features with nulls":   {k: v for k, v in null_counts.items() if v > 0} or "None",
        "DB rows inserted":      inserted,
        "Output CSV":            str(csv_path),
        "Status":                "COMPLETE",
    }
    write_module_summary("module5_features", summary, config)

    print("\n" + "="*55)
    print("  MODULE 5 — FEATURE ENGINEERING REPORT")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ 5 figures saved to outputs/figures/")
    print(f"  ✓ master_features.csv → data/processed/")
    print(f"  ✓ engineered_features table populated in SQLite")
    print("="*55)

    return df


if __name__ == "__main__":
    run()
