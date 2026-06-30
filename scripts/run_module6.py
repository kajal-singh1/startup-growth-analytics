"""
Module 6 — Feature Engineering & Data Cleaning
================================================
Objective:
    Take the raw master_dataset.csv (150 rows × 28 cols) and produce a
    clean, fully-engineered feature matrix ready for advanced ML, causal
    inference, clustering, and forecasting.

Steps:
    1. Missing value treatment (YoY lag rows filled with country median)
    2. Outlier detection & capping (IQR method per numeric column)
    3. Feature scaling  (StandardScaler — stored for inverse transforms)
    4. 12 new engineered features (composite scores, ratios, flags)
    5. One-hot encode top_sector and pandemic_period
    6. Save clean dataset + scaler + feature list to disk
    7. 8 diagnostic figures

Outputs:
    data/processed/master_features.csv   ← 150 rows × ~45 cols
    data/processed/feature_list.txt
    models/scaler.joblib
    data/outputs/figures/module6/*.png   ← 8 figures
    data/outputs/reports/module6_feature_report.txt

Usage:
    python scripts/run_module6.py
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
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, get_project_root, get_db_connection

logger = get_logger("module6_features")

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0D1B2A"
PANEL_BG = "#1A2A3A"
TEAL     = "#00C9B1"
CYAN     = "#00BFFF"
GOLD     = "#FFD700"
CORAL    = "#FF6B6B"
WHITE    = "#E8EEF4"
GRID     = "#2A3F55"

OUT_DIR  = get_project_root() / "data/outputs/figures/module6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_style(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]
    for ax in np.array(axes).flatten():
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.7)


def save_fig(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved {name}.png")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & inspect
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    path = get_project_root() / "data/processed/master_dataset.csv"
    df   = pd.read_csv(path)
    logger.info(f"Loaded master dataset: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Missing value treatment
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    # YoY lag columns: fill with country-level median (first year per country has no lag)
    for col in ["startup_growth_yoy", "funding_growth_yoy"]:
        df[col] = df.groupby("country_code")[col].transform(
            lambda x: x.fillna(x.median())
        )

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Outlier detection & capping (IQR, per column globally)
# ─────────────────────────────────────────────────────────────────────────────
OUTLIER_COLS = [
    "gdp_usd_bn", "total_funding_usd_mn", "startup_count",
    "startup_density", "funding_intensity", "unicorn_rate",
    "startup_growth_yoy", "funding_growth_yoy",
]

def cap_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df   = df.copy()
    caps = {}
    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        total_capped = 0
        bounds_log = []
        # Compute IQR bounds PER COUNTRY, not globally — startup ecosystems
        # vary by orders of magnitude across countries, so a single global
        # ceiling clips every large country's high-growth years down to
        # the same flat value. Capping within each country preserves real
        # year-to-year variation while still controlling for genuine
        # within-country outliers.
        def cap_group(group):
            nonlocal total_capped
            q1, q3 = group[col].quantile(0.25), group[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return group
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((group[col] < lo) | (group[col] > hi)).sum()
            total_capped += int(n_out)
            group[col] = group[col].clip(lo, hi)
            return group

        df = df.groupby("country_code", group_keys=False).apply(cap_group)
        caps[col] = {"capped_rows": total_capped, "method": "per-country IQR"}

    total_capped = sum(v["capped_rows"] for v in caps.values())
    logger.info(f"Outlier capping: {total_capped} values capped across {len(caps)} columns (per-country)")
    return df, caps


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Feature Engineering (12 new features)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Innovation Score (composite: R&D + internet + ease of doing business)
    #    Normalize each component to 0-1 range first
    def minmax(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

    df["innovation_score"] = (
        0.40 * minmax(df["rd_expenditure_pct_gdp"]) +
        0.35 * minmax(df["internet_penetration"]) +
        0.25 * minmax(100 - df["ease_of_doing_business_rank"])  # lower rank = better
    ).round(4)

    # 2. Digital Readiness Index
    df["digital_readiness"] = (
        0.50 * minmax(df["internet_penetration"]) +
        0.30 * minmax(df["rd_expenditure_pct_gdp"]) +
        0.20 * minmax(df["gdp_per_capita"])
    ).round(4)

    # 3. Investment Efficiency (unicorns per billion USD invested)
    df["investment_efficiency"] = (
        df["num_unicorns"] / (df["total_funding_usd_mn"] / 1000 + 1e-6)
    ).round(6)

    # 4. Economic Stability Score (inverse of unemployment + positive GDP growth)
    df["economic_stability"] = (
        0.60 * minmax(df["gdp_growth_rate"] + 10) +   # shift so all positive
        0.40 * minmax(20 - df["unemployment_rate"])    # lower unemployment = better
    ).round(4)

    # 5. Startup Momentum (3-year rolling average startup growth, per country)
    df = df.sort_values(["country_code", "year"])
    df["startup_momentum"] = (
        df.groupby("country_code")["startup_growth_yoy"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    ).round(4)

    # 6. Funding Momentum (3-year rolling average funding growth, per country)
    df["funding_momentum"] = (
        df.groupby("country_code")["funding_growth_yoy"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    ).round(4)

    # 7. GDP per startup (economic output per startup unit)
    df["gdp_per_startup"] = (
        (df["gdp_usd_bn"] * 1e9) / (df["startup_count"] + 1)
    ).round(2)

    # 8. Unicorn pipeline ratio (unicorns relative to startup base)
    df["unicorn_pipeline"] = (
        df["cumulative_unicorns"] / (df["startup_count"] + 1) * 1000
    ).round(4)

    # 9. Post-pandemic recovery flag (post period & growth above pre-pandemic median)
    pre_median_growth = df[df["pandemic_period"] == "pre"]["startup_growth_yoy"].median()
    df["strong_recovery"] = (
        (df["pandemic_period"] == "post") &
        (df["startup_growth_yoy"] > pre_median_growth)
    ).astype(int)

    # 10. High innovation flag (above median innovation score)
    df["high_innovation"] = (
        df["innovation_score"] > df["innovation_score"].median()
    ).astype(int)

    # 11. Funding per deal (avg deal size proxy)
    df["funding_per_deal"] = (
        df["total_funding_usd_mn"] / (df["num_deals"] + 1)
    ).round(4)

    # 12. Year index (0-based, for trend models)
    df["year_idx"] = df["year"] - df["year"].min()

    logger.info("Engineered 12 new features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Encode categoricals
# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-hot encode top_sector
    sector_dummies = pd.get_dummies(df["top_sector"], prefix="sector", drop_first=False)
    df = pd.concat([df, sector_dummies], axis=1)

    # Encode pandemic_period as ordered integer
    period_map = {"pre": 0, "during": 1, "post": 2}
    df["period_enc"] = df["pandemic_period"].map(period_map)

    # One-hot encode pandemic_period
    period_dummies = pd.get_dummies(df["pandemic_period"], prefix="period", drop_first=False)
    df = pd.concat([df, period_dummies], axis=1)

    logger.info(f"Encoded categoricals — shape now {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Scale numeric features
# ─────────────────────────────────────────────────────────────────────────────
SCALE_COLS = [
    "gdp_usd_bn", "gdp_per_capita", "gdp_growth_rate",
    "internet_penetration", "unemployment_rate", "population_mn",
    "rd_expenditure_pct_gdp", "startup_count", "total_funding_usd_mn",
    "num_deals", "num_unicorns", "startup_density", "funding_intensity",
    "startup_growth_yoy", "funding_growth_yoy", "innovation_score",
    "digital_readiness", "investment_efficiency", "economic_stability",
    "startup_momentum", "funding_momentum", "funding_per_deal",
]

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list]:
    df = df.copy()
    cols_present = [c for c in SCALE_COLS if c in df.columns]

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[cols_present])
    scaled_cols   = [f"{c}_scaled" for c in cols_present]
    df_scaled     = pd.DataFrame(scaled_values, columns=scaled_cols, index=df.index)
    df = pd.concat([df, df_scaled], axis=1)

    logger.info(f"Scaled {len(cols_present)} features")
    return df, scaler, cols_present


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def fig01_missing_heatmap(df_raw, df_clean, saved):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    apply_style(fig, axes)

    for ax, data, title in zip(axes,
        [df_raw[["startup_growth_yoy", "funding_growth_yoy"]],
         df_clean[["startup_growth_yoy", "funding_growth_yoy"]]],
        ["Before: Missing Values", "After: Missing Values Filled"]
    ):
        miss = data.isnull().astype(int)
        sns.heatmap(miss.T, ax=ax, cmap=["#1A2A3A", CORAL],
                    cbar=False, linewidths=0.1, linecolor=DARK_BG)
        ax.set_title(title, color=WHITE, fontsize=11)
        ax.set_xlabel("Row index")
        plt.setp(ax.get_yticklabels(), color=WHITE)
        plt.setp(ax.get_xticklabels(), color=WHITE)

    fig.suptitle("Missing Value Treatment — YoY Lag Columns", color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "01_missing_value_treatment"))


def fig02_outlier_before_after(df_raw, df_clean, saved):
    cols = ["startup_growth_yoy", "funding_growth_yoy", "startup_density", "funding_intensity"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    apply_style(fig, axes)

    for i, col in enumerate(cols):
        for j, (data, label, color) in enumerate([
            (df_raw, "Before capping", CORAL),
            (df_clean, "After capping", TEAL)
        ]):
            ax = axes[j][i]
            ax.hist(data[col].dropna(), bins=20, color=color, alpha=0.8, edgecolor=DARK_BG)
            ax.set_title(f"{col}\n{label}", fontsize=8, color=WHITE)

    fig.suptitle("Outlier Capping (IQR Method) — Before vs After", color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "02_outlier_capping"))


def fig03_engineered_distributions(df, saved):
    new_features = [
        "innovation_score", "digital_readiness", "economic_stability",
        "investment_efficiency", "startup_momentum", "funding_momentum",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    apply_style(fig, axes)

    for ax, feat in zip(axes.flatten(), new_features):
        data = df[feat].dropna()
        ax.hist(data, bins=20, color=TEAL, alpha=0.8, edgecolor=DARK_BG)
        ax.axvline(data.mean(), color=GOLD, lw=2, linestyle="--", label=f"Mean={data.mean():.2f}")
        ax.axvline(data.median(), color=CORAL, lw=2, linestyle=":", label=f"Median={data.median():.2f}")
        ax.set_title(feat.replace("_", " ").title(), color=WHITE, fontsize=10)
        ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)

    fig.suptitle("Distribution of 6 Engineered Features", color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "03_engineered_feature_distributions"))


def fig04_innovation_score_by_country(df, saved):
    df23 = df[df["year"] == 2023].sort_values("innovation_score", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_style(fig, ax)

    colors = [GOLD if v > df23["innovation_score"].median() else TEAL
              for v in df23["innovation_score"]]
    bars = ax.barh(df23["country_name"], df23["innovation_score"],
                   color=colors, alpha=0.85, edgecolor=DARK_BG)

    for bar, val in zip(bars, df23["innovation_score"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color=WHITE, fontsize=9)

    ax.axvline(df23["innovation_score"].median(), color=CORAL,
               lw=1.5, linestyle="--", label="Median")
    ax.set_title("Innovation Score by Country (2023)\n0.4×R&D + 0.35×Internet + 0.25×Ease of Business",
                 color=WHITE, fontsize=11)
    ax.set_xlabel("Innovation Score (0–1)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)
    fig.tight_layout()
    saved.append(save_fig(fig, "04_innovation_score_by_country"))


def fig05_digital_readiness_trend(df, saved):
    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    top5 = df[df["year"] == 2023].nlargest(5, "digital_readiness")["country_name"].tolist()
    colors = [TEAL, CYAN, GOLD, CORAL, "#B39DDB"]

    for country, color in zip(top5, colors):
        grp = df[df["country_name"] == country].sort_values("year")
        ax.plot(grp["year"], grp["digital_readiness"],
                color=color, lw=2.5, marker="o", markersize=5, label=country)

    ax.axvspan(2020, 2021.5, color=CORAL, alpha=0.1, label="Pandemic period")
    ax.set_title("Digital Readiness Index — Top 5 Countries (2015–2024)", color=WHITE, fontsize=12)
    ax.set_xlabel("Year"); ax.set_ylabel("Digital Readiness Score (0–1)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)
    fig.tight_layout()
    saved.append(save_fig(fig, "05_digital_readiness_trend"))


def fig06_correlation_engineered(df, saved):
    eng_cols = [
        "innovation_score", "digital_readiness", "economic_stability",
        "investment_efficiency", "startup_momentum", "funding_momentum",
        "startup_growth_yoy", "funding_intensity", "startup_density",
    ]
    present = [c for c in eng_cols if c in df.columns]
    corr = df[present].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    apply_style(fig, ax)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, linecolor=DARK_BG,
                xticklabels=[c.replace("_", "\n") for c in present],
                yticklabels=[c.replace("_", "\n") for c in present],
                ax=ax, cbar_kws={"shrink": 0.7})

    ax.set_title("Correlation Matrix — Engineered Features", color=WHITE, fontsize=13)
    plt.setp(ax.get_xticklabels(), color=WHITE, fontsize=8)
    plt.setp(ax.get_yticklabels(), color=WHITE, fontsize=8)
    cbar = ax.collections[0].colorbar
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
    fig.tight_layout()
    saved.append(save_fig(fig, "06_engineered_feature_correlations"))


def fig07_scaling_comparison(df, saved):
    cols_to_show = ["startup_count", "total_funding_usd_mn", "gdp_usd_bn"]
    scaled_cols  = [f"{c}_scaled" for c in cols_to_show]
    present      = [(c, s) for c, s in zip(cols_to_show, scaled_cols) if s in df.columns]

    fig, axes = plt.subplots(2, len(present), figsize=(14, 7))
    apply_style(fig, axes)

    for i, (raw_col, scaled_col) in enumerate(present):
        # Raw
        axes[0][i].hist(df[raw_col].dropna(), bins=20, color=CORAL, alpha=0.8, edgecolor=DARK_BG)
        axes[0][i].set_title(f"{raw_col}\n(raw)", color=WHITE, fontsize=9)
        # Scaled
        axes[1][i].hist(df[scaled_col].dropna(), bins=20, color=TEAL, alpha=0.8, edgecolor=DARK_BG)
        axes[1][i].set_title(f"{scaled_col}\n(standardized)", color=WHITE, fontsize=9)

    fig.suptitle("Feature Scaling — Raw vs StandardScaler", color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "07_feature_scaling"))


def fig08_feature_summary_table(df, caps, saved):
    new_features = [
        "innovation_score", "digital_readiness", "economic_stability",
        "investment_efficiency", "startup_momentum", "funding_momentum",
        "gdp_per_startup", "unicorn_pipeline", "funding_per_deal",
        "strong_recovery", "high_innovation", "year_idx",
    ]
    present = [f for f in new_features if f in df.columns]

    rows = []
    for feat in present:
        s = df[feat].dropna()
        rows.append([
            feat,
            f"{s.mean():.3f}",
            f"{s.std():.3f}",
            f"{s.min():.3f}",
            f"{s.max():.3f}",
            str(s.isnull().sum()),
        ])

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Feature", "Mean", "Std", "Min", "Max", "Nulls"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#0A3D5C" if r == 0 else PANEL_BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor(GRID)

    ax.set_title("Engineered Features — Summary Statistics", color=WHITE,
                 fontsize=13, pad=20, loc="left")
    fig.tight_layout()
    saved.append(save_fig(fig, "08_feature_summary_table"))


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(df_raw, df_final, caps, feature_list, scaler_path):
    path = get_project_root() / "data/outputs/reports/module6_feature_report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODULE 6 — FEATURE ENGINEERING & DATA CLEANING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write(f"  Input  : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols\n")
        f.write(f"  Output : {df_final.shape[0]} rows x {df_final.shape[1]} cols\n\n")

        f.write("MISSING VALUE TREATMENT\n")
        f.write("  startup_growth_yoy  : filled with country-level median\n")
        f.write("  funding_growth_yoy  : filled with country-level median\n")
        f.write(f"  Total missing after : {df_final.isnull().sum().sum()}\n\n")

        f.write("OUTLIER CAPPING (IQR x1.5, computed per-country)\n")
        for col, info in caps.items():
            f.write(f"  {col:<35} capped={info['capped_rows']} rows | "
                    f"method={info.get('method', 'per-country IQR')}\n")

        f.write("\nENGINEERED FEATURES (12 new)\n")
        eng = [
            ("innovation_score",      "0.4*R&D + 0.35*Internet + 0.25*EaseOfBusiness (normalized)"),
            ("digital_readiness",     "0.5*Internet + 0.3*R&D + 0.2*GDPperCapita (normalized)"),
            ("investment_efficiency", "Unicorns per billion USD invested"),
            ("economic_stability",    "0.6*GDP_growth + 0.4*(1-unemployment) (normalized)"),
            ("startup_momentum",      "3-year rolling avg startup growth YoY per country"),
            ("funding_momentum",      "3-year rolling avg funding growth YoY per country"),
            ("gdp_per_startup",       "GDP (USD) / startup count"),
            ("unicorn_pipeline",      "Cumulative unicorns per 1000 startups"),
            ("strong_recovery",       "Flag: post-pandemic + growth > pre-pandemic median"),
            ("high_innovation",       "Flag: innovation_score > median"),
            ("funding_per_deal",      "Total funding / number of deals"),
            ("year_idx",              "Year offset from 2015 (0-based)"),
        ]
        for feat, desc in eng:
            f.write(f"  {feat:<30} {desc}\n")

        f.write(f"\nSCALING\n")
        f.write(f"  Method  : StandardScaler (zero mean, unit variance)\n")
        f.write(f"  Columns : {len(feature_list)}\n")
        f.write(f"  Saved   : {scaler_path}\n")

        f.write(f"\nFEATURE LIST ({len(feature_list)} ML-ready features)\n")
        for feat in feature_list:
            f.write(f"  {feat}\n")

    logger.info("Report saved to data/outputs/reports/module6_feature_report.txt")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    logger.info("=" * 60)
    logger.info("MODULE 6: FEATURE ENGINEERING & DATA CLEANING — START")
    logger.info("=" * 60)

    saved = []

    # 1. Load
    df_raw = load_data()
    print(f"\n  Input  : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
    print(f"  Missing: {df_raw.isnull().sum().sum()} cells")

    # 2. Missing values
    df = handle_missing(df_raw)
    fig01_missing_heatmap(df_raw, df, saved)

    # 3. Outlier capping
    df_before_cap = df.copy()
    df, caps = cap_outliers(df)
    fig02_outlier_before_after(df_before_cap, df, saved)

    # 4. Feature engineering
    df = engineer_features(df)
    fig03_engineered_distributions(df, saved)
    fig04_innovation_score_by_country(df, saved)
    fig05_digital_readiness_trend(df, saved)

    # 5. Encode categoricals
    df = encode_categoricals(df)

    # 6. Scale
    df, scaler, scale_cols = scale_features(df)
    fig06_correlation_engineered(df, saved)
    fig07_scaling_comparison(df, saved)
    fig08_feature_summary_table(df, caps, saved)

    # Save master_features.csv
    out_path = get_project_root() / "data/processed/master_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved master_features.csv: {df.shape}")

    # Save feature list
    ml_features = (
        [c for c in df.columns if c.endswith("_scaled")] +
        ["innovation_score", "digital_readiness", "economic_stability",
         "investment_efficiency", "startup_momentum", "funding_momentum",
         "gdp_per_startup", "unicorn_pipeline", "funding_per_deal",
         "strong_recovery", "high_innovation", "year_idx", "period_enc",
         "is_pandemic", "is_post_pandemic", "is_partial"] +
        [c for c in df.columns if c.startswith("sector_") or c.startswith("period_")]
    )
    ml_features = [f for f in ml_features if f in df.columns]

    feat_path = get_project_root() / "data/processed/feature_list.txt"
    with open(feat_path, "w") as f:
        f.write("\n".join(ml_features))
    logger.info(f"Feature list: {len(ml_features)} features saved")

    # Save scaler
    scaler_path = get_project_root() / "models/scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to models/scaler.joblib")

    # Save to SQLite
    try:
        conn = get_db_connection()
        df.to_sql("engineered_features", conn, if_exists="replace", index=False)
        conn.close()
        logger.info("Saved to SQLite: engineered_features table")
    except Exception as e:
        logger.warning(f"SQLite save skipped: {e}")

    # Write report
    write_report(df_raw, df, caps, ml_features, scaler_path)

    logger.info("=" * 60)
    logger.info(f"MODULE 6 COMPLETE — {len(saved)} figures + 1 report")
    logger.info(f"Location: {OUT_DIR}")
    for p in saved:
        logger.info(f"  {Path(p).name}")

    print(f"\n  Output : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  New features : 12 engineered")
    print(f"  ML features  : {len(ml_features)}")
    print(f"  Missing after: {df.isnull().sum().sum()}")
    print(f"\n{'='*60}")
    print(f"MODULE 6 COMPLETE — {len(saved)} figures + 1 report")
    print(f"  master_features.csv saved")
    print(f"  feature_list.txt saved")
    print(f"  models/scaler.joblib saved")
    print(f"{'='*60}")
    print(f"\nNext: python scripts\\run_module7.py  (Causal Inference)")

    return df, ml_features


if __name__ == "__main__":
    run()
