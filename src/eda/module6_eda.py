"""
module6_eda.py — Exploratory Data Analysis
============================================

OBJECTIVE
---------
Understand the structure, distributions, trends, and relationships
in the clean feature dataset before any modelling begins.

WHY EDA IS NEEDED
-----------------
Models learn from patterns in data. If you don't know what patterns
exist, you can't validate whether the model learned the right things.
EDA answers:
  - How does startup growth vary across countries and years?
  - Which economic indicators correlate most with startup growth?
  - Did the pandemic cause a structural break in the data?
  - Are there country clusters with similar growth profiles?

FIGURES PRODUCED (12)
----------------------
 1. Startup count trend — all countries over time
 2. Top 5 vs bottom 5 countries by mean growth rate
 3. Pandemic impact — pre vs post mean startup count
 4. Correlation heatmap — all numeric features vs target
 5. Scatter: GDP per capita vs startup count growth rate
 6. Scatter: Internet penetration vs startup count
 7. Box plot: growth rate distribution by pandemic period
 8. Funding trend — total funding USD over time (top 5)
 9. Innovation score vs growth rate (scatter + regression line)
10. Economic momentum — heatmap country × year
11. Pairplot — 4 key features coloured by pandemic period
12. Year-on-year growth rate heatmap — country × year

INPUTS
------
- data/processed/master_features.csv

OUTPUTS
-------
- outputs/figures/module6_01_*.png … module6_12_*.png
- outputs/reports/module6_eda_report.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (load_config, setup_logging, save_figure,
                   load_dataframe, write_module_summary, set_seeds)

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE   = sns.color_palette("tab20", 15)
PRE_COL   = "#3498db"
POST_COL  = "#e74c3c"
TARGET    = "startup_count_growth_rate"


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fig1_startup_trend(df, config, logger):
    """Startup count over time for all 15 countries."""
    fig, ax = plt.subplots(figsize=(14, 6))
    countries = df["country"].unique()
    colors = cm.tab20(np.linspace(0, 1, len(countries)))
    for i, country in enumerate(sorted(countries)):
        sub = df[df["country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["startup_count"],
                marker="o", linewidth=1.8, markersize=4,
                color=colors[i], label=country)
    ax.axvspan(2019.5, 2021.5, alpha=0.12, color="red", label="COVID-19 period")
    ax.set_title("Startup Count Trend by Country (2015–2023)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Startup Count")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()
    p = save_figure(fig, "module6_01_startup_trend.png", config)
    logger.info(f"Fig 1 saved: {p}")


def fig2_top_bottom_countries(df, config, logger):
    """Top 5 vs bottom 5 countries by mean startup growth rate."""
    mean_growth = (df.groupby("country")[TARGET]
                   .mean().dropna().sort_values(ascending=False))
    top5    = mean_growth.head(5)
    bottom5 = mean_growth.tail(5)
    combined = pd.concat([top5, bottom5])

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [PRE_COL if v >= 0 else POST_COL for v in combined.values]
    bars = ax.barh(combined.index, combined.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Top 5 vs Bottom 5 Countries — Mean Startup Growth Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Annual Growth Rate (%)")
    plt.tight_layout()
    p = save_figure(fig, "module6_02_top_bottom_countries.png", config)
    logger.info(f"Fig 2 saved: {p}")


def fig3_pandemic_impact(df, config, logger):
    """Pre vs post pandemic mean startup count per country."""
    pre  = df[df["pandemic_period"] == 0].groupby("country")["startup_count"].mean()
    post = df[df["pandemic_period"] == 1].groupby("country")["startup_count"].mean()
    comp = pd.DataFrame({"Pre-pandemic": pre, "Post-pandemic": post}).dropna().sort_values("Pre-pandemic")

    x = np.arange(len(comp))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, comp["Pre-pandemic"],  width=w, label="Pre-pandemic (2015–2019)",  color=PRE_COL,  alpha=0.85)
    ax.bar(x + w/2, comp["Post-pandemic"], width=w, label="Post-pandemic (2020–2023)", color=POST_COL, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(comp.index, rotation=45, ha="right")
    ax.set_title("Pandemic Impact — Mean Startup Count: Pre vs Post COVID-19", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Startup Count"); ax.legend()
    plt.tight_layout()
    p = save_figure(fig, "module6_03_pandemic_impact.png", config)
    logger.info(f"Fig 3 saved: {p}")


def fig4_correlation_heatmap(df, config, logger):
    """Correlation heatmap: all numeric features vs target."""
    raw_cols = [
        "gdp_usd", "gdp_per_capita_usd", "gdp_growth_rate",
        "internet_penetration_pct", "unemployment_rate",
        "startup_count", "total_funding_usd", "venture_capital_usd",
        "startup_density", "funding_per_startup_mn", "innovation_score",
        "digital_readiness_score", "investment_efficiency",
        "economic_momentum", "pandemic_interaction", TARGET
    ]
    cols = [c for c in raw_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, square=True,
                linewidths=0.4, ax=ax, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap — All Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = save_figure(fig, "module6_04_correlation_heatmap.png", config)
    logger.info(f"Fig 4 saved: {p}")


def fig5_gdp_vs_growth(df, config, logger):
    """Scatter: GDP per capita vs startup growth rate, coloured by pandemic."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for period, color, label in [(0, PRE_COL, "Pre-pandemic"), (1, POST_COL, "Post-pandemic")]:
        sub = df[df["pandemic_period"] == period]
        ax.scatter(sub["gdp_per_capita_usd"] / 1000, sub[TARGET],
                   color=color, alpha=0.65, s=60, label=label, edgecolors="white")
    # Regression line on full data
    x = df["gdp_per_capita_usd"].dropna() / 1000
    y = df[TARGET].dropna()
    idx = x.index.intersection(y.index)
    slope, intercept, r, p_val, _ = stats.linregress(x[idx], y[idx])
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=1.5,
            label=f"Regression (r={r:.2f}, p={p_val:.3f})")
    ax.set_title("GDP per Capita vs Startup Growth Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("GDP per Capita ($000s)"); ax.set_ylabel("Startup Growth Rate (%)")
    ax.legend(); plt.tight_layout()
    p = save_figure(fig, "module6_05_gdp_vs_growth.png", config)
    logger.info(f"Fig 5 saved: {p}")


def fig6_internet_vs_startups(df, config, logger):
    """Scatter: Internet penetration vs startup count."""
    fig, ax = plt.subplots(figsize=(10, 6))
    countries = df["country"].unique()
    colors = cm.tab20(np.linspace(0, 1, len(countries)))
    for i, country in enumerate(sorted(countries)):
        sub = df[df["country"] == country]
        ax.scatter(sub["internet_penetration_pct"], sub["startup_count"],
                   color=colors[i], alpha=0.7, s=55, label=country)
    ax.set_title("Internet Penetration vs Startup Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Internet Penetration (%)"); ax.set_ylabel("Startup Count")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()
    p = save_figure(fig, "module6_06_internet_vs_startups.png", config)
    logger.info(f"Fig 6 saved: {p}")


def fig7_growth_by_pandemic(df, config, logger):
    """Box plot: startup growth rate distribution by pandemic period."""
    df_plot = df.copy()
    df_plot["Period"] = df_plot["pandemic_period"].map(
        {0: "Pre-pandemic\n(2015–2019)", 1: "Post-pandemic\n(2020–2023)"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot
    sns.boxplot(data=df_plot, x="Period", y=TARGET,
                palette=[PRE_COL, POST_COL], ax=axes[0])
    axes[0].set_title("Growth Rate Distribution by Period", fontsize=12)
    axes[0].set_ylabel("Startup Growth Rate (%)")
    axes[0].set_xlabel("")

    # Violin plot
    sns.violinplot(data=df_plot, x="Period", y=TARGET,
                   palette=[PRE_COL, POST_COL], ax=axes[1], inner="quartile")
    axes[1].set_title("Growth Rate Density by Period", fontsize=12)
    axes[1].set_ylabel(""); axes[1].set_xlabel("")

    fig.suptitle("Startup Growth Rate: Pre vs Post Pandemic", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = save_figure(fig, "module6_07_growth_by_pandemic.png", config)
    logger.info(f"Fig 7 saved: {p}")


def fig8_funding_trend(df, config, logger):
    """Total funding trend over time — top 5 countries by total funding."""
    top5 = (df.groupby("country")["total_funding_usd"].sum()
              .sort_values(ascending=False).head(5).index)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [PRE_COL, POST_COL, "#2ecc71", "#f39c12", "#9b59b6"]
    for i, country in enumerate(top5):
        sub = df[df["country"] == country].sort_values("year")
        funding_bn = sub["total_funding_usd"] / 1e9
        ax.plot(sub["year"], funding_bn, marker="o", linewidth=2,
                color=colors[i], label=country)
    ax.axvspan(2019.5, 2021.5, alpha=0.1, color="red", label="COVID-19")
    ax.set_title("Total Startup Funding Trend — Top 5 Countries ($B)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Funding ($B)")
    ax.legend(); plt.tight_layout()
    p = save_figure(fig, "module6_08_funding_trend.png", config)
    logger.info(f"Fig 8 saved: {p}")


def fig9_innovation_vs_growth(df, config, logger):
    """Innovation score vs startup growth rate with regression line."""
    x = df["innovation_score"]
    y = df[TARGET]
    idx = x.dropna().index.intersection(y.dropna().index)
    slope, intercept, r, p_val, _ = stats.linregress(x[idx], y[idx])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x[idx], y[idx], c=df.loc[idx, "pandemic_period"],
               cmap="coolwarm", alpha=0.7, s=60, edgecolors="white")
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=1.8,
            label=f"r = {r:.3f},  p = {p_val:.3f}")
    ax.set_title("Innovation Score vs Startup Growth Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("Innovation Score (0–1)"); ax.set_ylabel("Startup Growth Rate (%)")
    ax.legend()
    # Add colourbar legend
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Pre-pandemic", "Post-pandemic"])
    plt.tight_layout()
    p = save_figure(fig, "module6_09_innovation_vs_growth.png", config)
    logger.info(f"Fig 9 saved: {p}")


def fig10_momentum_heatmap(df, config, logger):
    """Economic momentum heatmap — country × year."""
    pivot = df.pivot_table(index="country", columns="year",
                           values="economic_momentum", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("Economic Momentum (GDP Growth − Unemployment) — Country × Year",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("")
    plt.tight_layout()
    p = save_figure(fig, "module6_10_momentum_heatmap.png", config)
    logger.info(f"Fig 10 saved: {p}")


def fig11_pairplot(df, config, logger):
    """Pairplot of 4 key features coloured by pandemic period."""
    cols = ["gdp_growth_rate", "internet_penetration_pct",
            "innovation_score", TARGET]
    cols = [c for c in cols if c in df.columns]
    plot_df = df[cols + ["pandemic_period"]].dropna().copy()
    plot_df["Period"] = plot_df["pandemic_period"].map(
        {0: "Pre-pandemic", 1: "Post-pandemic"}
    )
    g = sns.pairplot(plot_df[cols + ["Period"]], hue="Period",
                     palette={"Pre-pandemic": PRE_COL, "Post-pandemic": POST_COL},
                     plot_kws={"alpha": 0.6, "s": 30},
                     diag_kind="kde", corner=True)
    g.figure.suptitle("Pairplot — Key Features by Pandemic Period",
                       y=1.01, fontsize=13, fontweight="bold")
    p = save_figure(g.figure, "module6_11_pairplot.png", config)
    logger.info(f"Fig 11 saved: {p}")


def fig12_growth_heatmap(df, config, logger):
    """Year-on-year growth rate heatmap — country × year."""
    pivot = df.pivot_table(index="country", columns="year",
                           values=TARGET, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8},
                cbar_kws={"label": "Growth Rate (%)"})
    ax.set_title("Startup Count Growth Rate (%) — Country × Year",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("")
    plt.tight_layout()
    p = save_figure(fig, "module6_12_growth_heatmap.png", config)
    logger.info(f"Fig 12 saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_eda_stats(df, logger):
    """Compute key EDA statistics for the report."""
    stats_out = {}

    # Basic shape
    stats_out["rows"] = len(df)
    stats_out["columns"] = df.shape[1]
    stats_out["countries"] = df["country"].nunique()
    stats_out["years"] = f"{df['year'].min()} – {df['year'].max()}"

    # Target variable
    target = df[TARGET].dropna()
    stats_out["target_mean"]   = round(target.mean(), 2)
    stats_out["target_median"] = round(target.median(), 2)
    stats_out["target_std"]    = round(target.std(), 2)
    stats_out["target_min"]    = round(target.min(), 2)
    stats_out["target_max"]    = round(target.max(), 2)

    # Pandemic comparison
    pre  = df[df["pandemic_period"] == 0][TARGET].dropna()
    post = df[df["pandemic_period"] == 1][TARGET].dropna()
    stats_out["pre_pandemic_mean_growth"]  = round(pre.mean(), 2)
    stats_out["post_pandemic_mean_growth"] = round(post.mean(), 2)

    # T-test: is pandemic effect significant?
    t_stat, p_val = stats.ttest_ind(pre, post)
    stats_out["pandemic_ttest_t"]  = round(t_stat, 4)
    stats_out["pandemic_ttest_p"]  = round(p_val, 4)
    stats_out["pandemic_effect_significant"] = "YES" if p_val < 0.05 else "NO"

    # Top correlations with target
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corrs = df[num_cols].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    stats_out["top_3_correlated_features"] = list(corrs.head(3).index)
    stats_out["top_3_correlation_values"]  = [round(v, 3) for v in corrs.head(3).values]

    # Best and worst growth countries
    mean_g = df.groupby("country")[TARGET].mean().dropna()
    stats_out["highest_growth_country"] = mean_g.idxmax()
    stats_out["lowest_growth_country"]  = mean_g.idxmin()
    stats_out["highest_growth_value"]   = round(mean_g.max(), 2)
    stats_out["lowest_growth_value"]    = round(mean_g.min(), 2)

    for k, v in stats_out.items():
        logger.info(f"  EDA stat — {k}: {v}")

    return stats_out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(config=None):
    if config is None:
        config = load_config()
    set_seeds(config)
    logger = setup_logging("module6_eda", config)

    logger.info("=" * 55)
    logger.info("MODULE 6 — EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 55)

    df = load_dataframe("master_features.csv", stage="processed", config=config)
    logger.info(f"Loaded: {df.shape}")

    # All 12 figures
    fig1_startup_trend(df, config, logger)
    fig2_top_bottom_countries(df, config, logger)
    fig3_pandemic_impact(df, config, logger)
    fig4_correlation_heatmap(df, config, logger)
    fig5_gdp_vs_growth(df, config, logger)
    fig6_internet_vs_startups(df, config, logger)
    fig7_growth_by_pandemic(df, config, logger)
    fig8_funding_trend(df, config, logger)
    fig9_innovation_vs_growth(df, config, logger)
    fig10_momentum_heatmap(df, config, logger)
    fig11_pairplot(df, config, logger)
    fig12_growth_heatmap(df, config, logger)

    # Statistics
    eda_stats = compute_eda_stats(df, logger)

    summary = {
        "Dataset shape":                   f"{df.shape[0]} rows × {df.shape[1]} cols",
        "Countries":                        eda_stats["countries"],
        "Years":                            eda_stats["years"],
        "Target mean growth rate (%)":      eda_stats["target_mean"],
        "Target std (%)":                   eda_stats["target_std"],
        "Target range (%)":                 f"{eda_stats['target_min']} to {eda_stats['target_max']}",
        "Pre-pandemic mean growth (%)":     eda_stats["pre_pandemic_mean_growth"],
        "Post-pandemic mean growth (%)":    eda_stats["post_pandemic_mean_growth"],
        "Pandemic effect significant":      eda_stats["pandemic_effect_significant"],
        "Pandemic t-test p-value":          eda_stats["pandemic_ttest_p"],
        "Top correlated features":          eda_stats["top_3_correlated_features"],
        "Highest growth country":           f"{eda_stats['highest_growth_country']} ({eda_stats['highest_growth_value']}%)",
        "Lowest growth country":            f"{eda_stats['lowest_growth_country']} ({eda_stats['lowest_growth_value']}%)",
        "Figures saved":                    12,
        "Status":                           "COMPLETE",
    }

    write_module_summary("module6_eda", summary, config)

    print("\n" + "="*55)
    print("  MODULE 6 — EDA REPORT")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ 12 figures saved to outputs/figures/")
    print(f"  ✓ Report saved to outputs/reports/")
    print("="*55)

    return df, eda_stats


if __name__ == "__main__":
    run()
