"""
Module 3 — Exploratory Data Analysis (EDA)
Generates 12 publication-quality figures covering:
  1.  Startup count time series (all countries)
  2.  Total funding time series (all countries)
  3.  GDP growth rate time series
  4.  Pandemic period box plots — startup count
  5.  Pandemic period box plots — funding
  6.  Correlation heatmap
  7.  Internet penetration vs startup density scatter
  8.  GDP per capita vs funding intensity scatter
  9.  Top sectors bar chart (2024)
  10. Unicorn production by country (cumulative)
  11. Startup growth YoY — pre/during/post comparison
  12. R&D expenditure vs startup density

Usage:
    python scripts/run_module3.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, get_project_root

logger = get_logger("module3_eda")

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0D1B2A"
PANEL_BG  = "#1A2A3A"
TEAL      = "#00C9B1"
CYAN      = "#00BFFF"
GOLD      = "#FFD700"
CORAL     = "#FF6B6B"
LAVENDER  = "#B39DDB"
WHITE     = "#E8EEF4"
GRID      = "#2A3F55"

COUNTRY_COLORS = {
    "United States":  "#00C9B1",
    "China":          "#FF6B6B",
    "India":          "#FFD700",
    "United Kingdom": "#00BFFF",
    "Germany":        "#B39DDB",
    "France":         "#FF9F43",
    "Singapore":      "#48DBFB",
    "Israel":         "#FF9FF3",
    "Canada":         "#54A0FF",
    "Australia":      "#5F27CD",
    "Brazil":         "#01CBC6",
    "South Korea":    "#EE5A24",
    "Netherlands":    "#C4E538",
    "Sweden":         "#FDA7DF",
    "Indonesia":      "#D980FA",
}

PERIOD_COLORS = {"pre": TEAL, "during": CORAL, "post": GOLD}

def apply_style(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.7)

def save_fig(fig, name, out_dir):
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved {name}.png")
    return path

OUT_DIR = get_project_root() / "data/outputs/figures/module3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run():
    df = pd.read_csv(get_project_root() / "data/processed/master_dataset.csv")
    df_full = df.copy()                        # includes 2024 partial
    df = df[df["is_partial"] == 0].copy()      # confirmed data only for most charts
    saved = []

    logger.info(f"EDA on {len(df_full)} rows ({len(df)} confirmed + {len(df_full)-len(df)} partial)")

    # ── 1. Startup Count Time Series ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    # Split into big-4 (prominent lines) and rest (thin)
    big4 = ["United States", "China", "India", "United Kingdom"]
    for country, grp in df_full.groupby("country_name"):
        confirmed = grp[grp["is_partial"] == 0]
        partial   = grp[grp["is_partial"] == 1]
        color = COUNTRY_COLORS.get(country, "#888888")
        lw    = 2.5 if country in big4 else 1.0
        alpha = 1.0 if country in big4 else 0.55
        ax.plot(confirmed["year"], confirmed["startup_count"] / 1000,
                color=color, lw=lw, alpha=alpha, label=country)
        if len(partial):
            ax.plot([confirmed["year"].iloc[-1], partial["year"].iloc[0]],
                    [confirmed["startup_count"].iloc[-1] / 1000,
                     partial["startup_count"].iloc[0] / 1000],
                    color=color, lw=lw, alpha=alpha, linestyle="--")
            ax.scatter(partial["year"], partial["startup_count"] / 1000,
                       color=color, s=40, zorder=5, marker="D")

    ax.axvspan(2020, 2021.5, color=CORAL, alpha=0.12, label="Pandemic period")
    ax.set_title("Startup Count by Country (2015–2024)", fontsize=14, pad=12, color=WHITE)
    ax.set_xlabel("Year"); ax.set_ylabel("Startups (thousands)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=4, loc="upper left",
              facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "01_startup_count_timeseries", OUT_DIR))

    # ── 2. Total Funding Time Series ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    for country, grp in df_full.groupby("country_name"):
        confirmed = grp[grp["is_partial"] == 0]
        partial   = grp[grp["is_partial"] == 1]
        color = COUNTRY_COLORS.get(country, "#888888")
        lw    = 2.5 if country in big4 else 1.0
        alpha = 1.0 if country in big4 else 0.55
        ax.plot(confirmed["year"], confirmed["total_funding_usd_mn"] / 1000,
                color=color, lw=lw, alpha=alpha, label=country)
        if len(partial):
            ax.plot([confirmed["year"].iloc[-1], partial["year"].iloc[0]],
                    [confirmed["total_funding_usd_mn"].iloc[-1] / 1000,
                     partial["total_funding_usd_mn"].iloc[0] / 1000],
                    color=color, lw=lw, alpha=alpha, linestyle="--")
            ax.scatter(partial["year"], partial["total_funding_usd_mn"] / 1000,
                       color=color, s=40, zorder=5, marker="D")

    ax.axvspan(2020, 2021.5, color=CORAL, alpha=0.12, label="Pandemic period")
    ax.set_title("Total VC Funding by Country (2015–2024, USD Billions)", fontsize=14, pad=12, color=WHITE)
    ax.set_xlabel("Year"); ax.set_ylabel("Funding (USD Billions)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=4, loc="upper left",
              facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "02_funding_timeseries", OUT_DIR))

    # ── 3. GDP Growth Rate Time Series ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    global_avg = df_full.groupby("year")["gdp_growth_rate"].mean()
    ax.fill_between(global_avg.index, global_avg.values, alpha=0.15, color=TEAL)
    ax.plot(global_avg.index, global_avg.values, color=TEAL, lw=2.5,
            label="Global average", zorder=5)
    ax.axhline(0, color=WHITE, lw=0.8, linestyle="--", alpha=0.5)
    ax.axvspan(2020, 2021.5, color=CORAL, alpha=0.12, label="Pandemic shock")

    for country in ["United States", "India", "China", "Germany"]:
        grp = df_full[df_full["country_name"] == country]
        ax.plot(grp["year"], grp["gdp_growth_rate"],
                color=COUNTRY_COLORS[country], lw=1.3, alpha=0.8,
                linestyle=":", label=country)

    ax.set_title("GDP Growth Rate — Global Average vs Key Economies (2015–2024)", fontsize=14, pad=12, color=WHITE)
    ax.set_xlabel("Year"); ax.set_ylabel("GDP Growth Rate (%)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "03_gdp_growth_timeseries", OUT_DIR))

    # ── 4. Box Plot — Startup Count by Period ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    apply_style(fig, axes)

    order = ["pre", "during", "post"]
    palette = [PERIOD_COLORS[p] for p in order]

    # Exclude US and China (outliers that dwarf others)
    df_no_outliers = df[~df["country_name"].isin(["United States", "China"])]

    for ax, (col, title, ylabel) in zip(axes, [
        ("startup_count", "Startup Count Distribution by Pandemic Period\n(excl. US & China)", "Startup Count"),
        ("total_funding_usd_mn", "Total Funding Distribution by Pandemic Period\n(excl. US & China)", "Funding (USD Mn)"),
    ]):
        bp = ax.boxplot(
            [df_no_outliers[df_no_outliers["pandemic_period"] == p][col].dropna() for p in order],
            labels=["Pre\n(2015-19)", "During\n(2020-21)", "Post\n(2022-24)"],
            patch_artist=True, notch=False, widths=0.5,
            medianprops=dict(color=WHITE, lw=2),
            whiskerprops=dict(color=WHITE, lw=1),
            capprops=dict(color=WHITE, lw=1.5),
            flierprops=dict(marker="o", color=CORAL, alpha=0.5, markersize=4),
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11, color=WHITE)
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    saved.append(save_fig(fig, "04_boxplot_period_comparison", OUT_DIR))

    # ── 5. Correlation Heatmap ────────────────────────────────────────────────
    num_cols = [
        "gdp_usd_bn", "gdp_per_capita", "gdp_growth_rate",
        "internet_penetration", "unemployment_rate", "rd_expenditure_pct_gdp",
        "startup_count", "total_funding_usd_mn", "num_deals",
        "num_unicorns", "startup_density", "funding_intensity",
        "startup_growth_yoy", "funding_growth_yoy",
    ]
    corr = df[num_cols].corr()
    labels = [c.replace("_", "\n") for c in num_cols]

    fig, ax = plt.subplots(figsize=(14, 11))
    apply_style(fig, ax)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.4, linecolor=DARK_BG,
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={"shrink": 0.7})

    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12, color=WHITE)
    ax.tick_params(axis="x", rotation=0, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.setp(ax.get_xticklabels(), color=WHITE)
    plt.setp(ax.get_yticklabels(), color=WHITE)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=WHITE, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)

    fig.tight_layout()
    saved.append(save_fig(fig, "05_correlation_heatmap", OUT_DIR))

    # ── 6. Internet Penetration vs Startup Density ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_style(fig, ax)

    df23 = df[df["year"] == 2023].copy()
    for _, row in df23.iterrows():
        color = COUNTRY_COLORS.get(row["country_name"], TEAL)
        ax.scatter(row["internet_penetration"], row["startup_density"],
                   color=color, s=row["total_funding_usd_mn"] / 800,
                   alpha=0.85, edgecolors=WHITE, lw=0.5, zorder=3)
        ax.annotate(row["country_name"].replace("United ", "U."),
                    (row["internet_penetration"], row["startup_density"]),
                    fontsize=8, color=WHITE, alpha=0.9,
                    xytext=(5, 5), textcoords="offset points")

    # Trend line
    x, y = df23["internet_penetration"].values, df23["startup_density"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, p(xline), color=GOLD, lw=1.5, linestyle="--", alpha=0.7, label="Trend")

    ax.set_title("Internet Penetration vs Startup Density (2023)\nBubble size = Total Funding", fontsize=13, pad=12, color=WHITE)
    ax.set_xlabel("Internet Penetration (% Population)")
    ax.set_ylabel("Startup Density (per million population)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "06_internet_vs_startup_density", OUT_DIR))

    # ── 7. GDP per Capita vs Funding Intensity ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_style(fig, ax)

    for _, row in df23.iterrows():
        color = COUNTRY_COLORS.get(row["country_name"], TEAL)
        ax.scatter(row["gdp_per_capita"] / 1000, row["funding_intensity"],
                   color=color, s=120, alpha=0.85, edgecolors=WHITE, lw=0.5, zorder=3)
        ax.annotate(row["country_name"].replace("United ", "U."),
                    (row["gdp_per_capita"] / 1000, row["funding_intensity"]),
                    fontsize=8, color=WHITE, alpha=0.9,
                    xytext=(5, 4), textcoords="offset points")

    ax.set_title("GDP per Capita vs Funding Intensity (2023)\nFunding Intensity = Total Funding / GDP", fontsize=13, pad=12, color=WHITE)
    ax.set_xlabel("GDP per Capita (USD thousands)")
    ax.set_ylabel("Funding Intensity (USD Mn per Bn GDP)")
    fig.tight_layout()
    saved.append(save_fig(fig, "07_gdp_vs_funding_intensity", OUT_DIR))

    # ── 8. Top Sectors (2023) ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    sector_counts = df[df["year"] == 2023]["top_sector"].value_counts()
    colors_bar = [TEAL, CYAN, GOLD, CORAL, LAVENDER, "#FF9F43", "#54A0FF", "#C4E538"]
    bars = ax.bar(sector_counts.index, sector_counts.values,
                  color=colors_bar[:len(sector_counts)], alpha=0.85, edgecolor=DARK_BG, lw=0.5)

    for bar, val in zip(bars, sector_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")

    ax.set_title("Dominant Startup Sector per Country (2023)", fontsize=13, pad=12, color=WHITE)
    ax.set_xlabel("Sector"); ax.set_ylabel("Number of Countries")
    ax.set_ylim(0, sector_counts.max() + 1.5)
    fig.tight_layout()
    saved.append(save_fig(fig, "08_top_sectors_2023", OUT_DIR))

    # ── 9. Cumulative Unicorns by Country ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    uni_2023 = df[df["year"] == 2023][["country_name", "cumulative_unicorns"]].sort_values(
        "cumulative_unicorns", ascending=True)

    colors_uni = [COUNTRY_COLORS.get(c, TEAL) for c in uni_2023["country_name"]]
    bars = ax.barh(uni_2023["country_name"], uni_2023["cumulative_unicorns"],
                   color=colors_uni, alpha=0.85, edgecolor=DARK_BG, lw=0.5)

    for bar, val in zip(bars, uni_2023["cumulative_unicorns"]):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=WHITE, fontsize=9)

    ax.set_title("Cumulative Unicorns by Country (as of 2023)", fontsize=13, pad=12, color=WHITE)
    ax.set_xlabel("Cumulative Unicorn Count")
    fig.tight_layout()
    saved.append(save_fig(fig, "09_cumulative_unicorns", OUT_DIR))

    # ── 10. Startup Growth YoY — Period Comparison ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_style(fig, ax)

    df_yoy = df.dropna(subset=["startup_growth_yoy"])
    period_yoy = df_yoy.groupby(["pandemic_period", "year"])["startup_growth_yoy"].mean().reset_index()

    for period, color in PERIOD_COLORS.items():
        sub = period_yoy[period_yoy["pandemic_period"] == period]
        ax.plot(sub["year"], sub["startup_growth_yoy"], color=color, lw=2.5,
                marker="o", markersize=6, label=period.capitalize())
        ax.fill_between(sub["year"], sub["startup_growth_yoy"], alpha=0.15, color=color)

    ax.axhline(0, color=WHITE, lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Average Startup Growth Rate YoY by Pandemic Period", fontsize=13, pad=12, color=WHITE)
    ax.set_xlabel("Year"); ax.set_ylabel("YoY Growth Rate (%)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "10_startup_growth_yoy", OUT_DIR))

    # ── 11. R&D Expenditure vs Startup Density ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_style(fig, ax)

    df23_rd = df23.dropna(subset=["rd_expenditure_pct_gdp", "startup_density"])
    for _, row in df23_rd.iterrows():
        color = COUNTRY_COLORS.get(row["country_name"], TEAL)
        ax.scatter(row["rd_expenditure_pct_gdp"], row["startup_density"],
                   color=color, s=120, alpha=0.85, edgecolors=WHITE, lw=0.5, zorder=3)
        ax.annotate(row["country_name"].replace("United ", "U."),
                    (row["rd_expenditure_pct_gdp"], row["startup_density"]),
                    fontsize=8, color=WHITE, alpha=0.9,
                    xytext=(5, 4), textcoords="offset points")

    x2 = df23_rd["rd_expenditure_pct_gdp"].values
    y2 = df23_rd["startup_density"].values
    z2 = np.polyfit(x2, y2, 1)
    p2 = np.poly1d(z2)
    xline2 = np.linspace(x2.min(), x2.max(), 100)
    ax.plot(xline2, p2(xline2), color=GOLD, lw=1.5, linestyle="--", alpha=0.7, label="Trend")

    ax.set_title("R&D Expenditure (% GDP) vs Startup Density (2023)", fontsize=13, pad=12, color=WHITE)
    ax.set_xlabel("R&D Expenditure (% of GDP)")
    ax.set_ylabel("Startup Density (per million population)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "11_rd_vs_startup_density", OUT_DIR))

    # ── 12. Funding Distribution — Pre vs Post Pandemic (Top 6 Countries) ────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    apply_style(fig, axes.flatten())

    top6 = ["United States", "China", "India", "United Kingdom", "Israel", "Germany"]

    for ax, country in zip(axes.flatten(), top6):
        grp = df[df["country_name"] == country].sort_values("year")
        color = COUNTRY_COLORS.get(country, TEAL)

        years = grp["year"].values
        funding = grp["total_funding_usd_mn"].values / 1000

        bar_colors = [
            CORAL if y in [2020, 2021] else
            GOLD  if y >= 2022 else
            TEAL
            for y in years
        ]
        ax.bar(years, funding, color=bar_colors, alpha=0.85, edgecolor=DARK_BG, lw=0.3)
        ax.set_title(country, fontsize=10, color=WHITE)
        ax.set_ylabel("Funding (USD Bn)", fontsize=8)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=TEAL,  label="Pre-pandemic (2015-19)"),
        Patch(facecolor=CORAL, label="Pandemic (2020-21)"),
        Patch(facecolor=GOLD,  label="Post-pandemic (2022-23)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=9,
               facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Funding Trajectory — Pre vs During vs Post Pandemic (Top 6 Countries)",
                 fontsize=14, color=WHITE, y=1.01)
    fig.tight_layout()
    saved.append(save_fig(fig, "12_funding_pre_post_pandemic", OUT_DIR))

    return saved


if __name__ == "__main__":
    saved = run()
    print(f"\n✓ MODULE 3 COMPLETE — {len(saved)} figures saved")
    print(f"  Location: data/outputs/figures/module3/")
    for p in saved:
        print(f"  {p.name}")
