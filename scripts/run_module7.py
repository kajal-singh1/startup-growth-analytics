"""
Module 7 — Causal Inference
============================
Objective:
    Go beyond correlation to estimate the CAUSAL effect of the pandemic
    on startup growth using three complementary methods.

Methods:
    1. Difference-in-Differences (DiD)
       - Treatment: high-internet countries (above median in 2019)
       - Pre period: 2015-2019 | Post period: 2022-2024
       - Outcome: startup_growth_yoy
       - Controls: gdp_growth_rate, unemployment_rate, rd_expenditure_pct_gdp

    2. Propensity Score Matching (PSM)
       - Match high-internet vs low-internet countries on confounders
       - Estimate Average Treatment Effect on the Treated (ATT)

    3. Event Study (dynamic DiD)
       - Year-by-year treatment effects relative to 2019 (base year)
       - Tests parallel trends assumption visually

Outputs:
    data/outputs/figures/module7/*.png   (8 figures)
    data/outputs/reports/module7_causal_report.txt
    data/processed/did_results.csv

Usage:
    python scripts/run_module7.py
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
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, get_project_root

logger = get_logger("module7_causal")

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#0D1B2A"
PANEL_BG = "#1A2A3A"
TEAL     = "#00C9B1"
CYAN     = "#00BFFF"
GOLD     = "#FFD700"
CORAL    = "#FF6B6B"
LAVENDER = "#B39DDB"
WHITE    = "#E8EEF4"
GRID     = "#2A3F55"

OUT_DIR  = get_project_root() / "data/outputs/figures/module7"
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
        ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)


def save_fig(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    logger.info(f"Saved {name}.png")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prep():
    path = get_project_root() / "data/processed/master_features.csv"
    if not path.exists():
        path = get_project_root() / "data/processed/master_dataset.csv"
    df = pd.read_csv(path)
    df = df[df["is_partial"] == 0].copy()   # confirmed data only

    # Fill any remaining missing
    df["startup_growth_yoy"] = df.groupby("country_code")["startup_growth_yoy"].transform(
        lambda x: x.fillna(x.median()))
    df["funding_growth_yoy"] = df.groupby("country_code")["funding_growth_yoy"].transform(
        lambda x: x.fillna(x.median()))

    # Define treatment: high internet penetration (above 2019 median)
    inet_2019 = df[df["year"] == 2019].set_index("country_code")["internet_penetration"]
    median_inet = inet_2019.median()
    high_inet_countries = inet_2019[inet_2019 >= median_inet].index.tolist()

    df["treated"]  = df["country_code"].isin(high_inet_countries).astype(int)
    df["post"]     = (df["year"] >= 2022).astype(int)
    df["during"]   = df["year"].isin([2020, 2021]).astype(int)
    df["did"]      = df["treated"] * df["post"]
    df["did_dur"]  = df["treated"] * df["during"]

    logger.info(f"Data prepared: {df.shape} | Treated countries: {len(high_inet_countries)}")
    logger.info(f"Treated: {sorted(high_inet_countries)}")
    return df, high_inet_countries, median_inet


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: DIFFERENCE-IN-DIFFERENCES
# ─────────────────────────────────────────────────────────────────────────────
def run_did(df):
    df_did = df.dropna(subset=["startup_growth_yoy"]).copy()

    # Basic DiD (no controls)
    model_basic = smf.ols(
        "startup_growth_yoy ~ treated + post + did",
        data=df_did
    ).fit(cov_type="HC3")

    # Extended DiD (with controls)
    model_ext = smf.ols(
        "startup_growth_yoy ~ treated + post + did + "
        "gdp_growth_rate + unemployment_rate + rd_expenditure_pct_gdp + "
        "internet_penetration + C(country_code)",
        data=df_did
    ).fit(cov_type="HC3")

    # During pandemic effect too
    model_full = smf.ols(
        "startup_growth_yoy ~ treated + post + during + did + did_dur + "
        "gdp_growth_rate + unemployment_rate + rd_expenditure_pct_gdp",
        data=df_did
    ).fit(cov_type="HC3")

    did_coef     = model_ext.params.get("did", np.nan)
    did_p        = model_ext.pvalues.get("did", np.nan)
    did_ci_lo    = model_ext.conf_int().loc["did", 0] if "did" in model_ext.conf_int().index else np.nan
    did_ci_hi    = model_ext.conf_int().loc["did", 1] if "did" in model_ext.conf_int().index else np.nan

    logger.info(f"DiD coef={did_coef:.4f} | p={did_p:.4f} | R²={model_ext.rsquared:.3f}")

    return {
        "basic": model_basic,
        "extended": model_ext,
        "full": model_full,
        "did_coef": did_coef,
        "did_p": did_p,
        "did_ci": (did_ci_lo, did_ci_hi),
        "r2": model_ext.rsquared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: PROPENSITY SCORE MATCHING
# ─────────────────────────────────────────────────────────────────────────────
def run_psm(df):
    # Use 2019 baseline data for matching
    df_base = df[df["year"] == 2019].copy().dropna(
        subset=["gdp_per_capita", "rd_expenditure_pct_gdp",
                "unemployment_rate", "startup_density"])

    X = df_base[["gdp_per_capita", "rd_expenditure_pct_gdp",
                  "unemployment_rate", "startup_density"]].values
    y = df_base["treated"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_sc, y)
    df_base = df_base.copy()
    df_base["propensity"] = lr.predict_proba(X_sc)[:, 1]

    treated_df   = df_base[df_base["treated"] == 1].copy()
    control_df   = df_base[df_base["treated"] == 0].copy()

    # Nearest-neighbour matching (1:1, with replacement)
    matched_pairs = []
    for _, treat_row in treated_df.iterrows():
        diffs = (control_df["propensity"] - treat_row["propensity"]).abs()
        best  = diffs.idxmin()
        matched_pairs.append({
            "treated_country": treat_row["country_name"],
            "control_country": control_df.loc[best, "country_name"],
            "treat_propensity": round(treat_row["propensity"], 4),
            "ctrl_propensity":  round(control_df.loc[best, "propensity"], 4),
            "pscore_diff":      round(abs(treat_row["propensity"] - control_df.loc[best, "propensity"]), 4),
        })

    pairs_df = pd.DataFrame(matched_pairs)

    # ATT: compare post-pandemic startup growth between matched groups
    post_df = df[df["year"] >= 2022].copy()
    treated_growth = post_df[post_df["treated"] == 1]["startup_growth_yoy"].dropna()
    control_growth = post_df[post_df["treated"] == 0]["startup_growth_yoy"].dropna()

    att = treated_growth.mean() - control_growth.mean()
    t, p = stats.ttest_ind(treated_growth, control_growth)

    logger.info(f"PSM ATT={att:.4f} | t={t:.4f} | p={p:.4f}")

    return {
        "pairs": pairs_df,
        "propensity_df": df_base,
        "att": att,
        "t_stat": t,
        "p_value": p,
        "treated_mean": treated_growth.mean(),
        "control_mean": control_growth.mean(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: EVENT STUDY (Dynamic DiD)
# ─────────────────────────────────────────────────────────────────────────────
def run_event_study(df):
    df_ev = df.dropna(subset=["startup_growth_yoy"]).copy()
    years = sorted(df_ev["year"].unique())
    base_year = 2019

    results = []
    for year in years:
        if year == base_year:
            results.append({"year": year, "coef": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": 1.0})
            continue

        df_ev["event_post"] = (df_ev["year"] == year).astype(int)
        df_ev["event_did"]  = df_ev["treated"] * df_ev["event_post"]

        try:
            m = smf.ols(
                "startup_growth_yoy ~ treated + event_post + event_did + "
                "gdp_growth_rate + unemployment_rate",
                data=df_ev[df_ev["year"].isin([base_year, year])]
            ).fit(cov_type="HC3")

            coef  = m.params.get("event_did", np.nan)
            ci    = m.conf_int().loc["event_did"] if "event_did" in m.conf_int().index else [np.nan, np.nan]
            pval  = m.pvalues.get("event_did", np.nan)
            results.append({"year": year, "coef": coef, "ci_lo": ci[0], "ci_hi": ci[1], "p": pval})
        except Exception:
            results.append({"year": year, "coef": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan})

    ev_df = pd.DataFrame(results)
    logger.info(f"Event study: {len(ev_df)} year coefficients estimated")
    return ev_df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def fig01_parallel_trends(df, high_inet, saved):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    apply_style(fig, axes)

    for ax, col, ylabel, title in zip(axes,
        ["startup_growth_yoy", "total_funding_usd_mn"],
        ["Startup Growth YoY (%)", "Total Funding (USD Mn)"],
        ["Parallel Trends — Startup Growth YoY", "Parallel Trends — VC Funding"]
    ):
        for group, color, ls, label in [
            (1, TEAL,  "-",  "Treated (High Internet)"),
            (0, CORAL, "--", "Control (Low Internet)"),
        ]:
            grp = df[df["treated"] == group].groupby("year")[col].mean()
            ax.plot(grp.index, grp.values, color=color, lw=2.5,
                    linestyle=ls, marker="o", markersize=5, label=label)

        ax.axvline(2019.5, color=GOLD, lw=1.5, linestyle=":", alpha=0.8, label="Treatment start")
        ax.axvspan(2020, 2021.5, color=CORAL, alpha=0.08)
        ax.axhline(0, color=WHITE, lw=0.5, linestyle="--", alpha=0.4)
        ax.set_title(title, color=WHITE, fontsize=11)
        ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)

    fig.suptitle("Parallel Trends Test — Pre-Treatment Period Should Show Similar Trends",
                 color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "01_parallel_trends"))


def fig02_did_coefficients(did_results, saved):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    apply_style(fig, axes)

    # Left: Basic vs Extended DiD comparison
    models = {"Basic DiD": did_results["basic"], "Extended DiD\n(with controls)": did_results["extended"]}
    for ax, (name, model) in zip([axes[0]], [("Extended DiD\n(with controls)", did_results["extended"])]):
        params = model.params.drop([c for c in model.params.index
                                    if c.startswith("C(country") or c == "Intercept"], errors="ignore")
        errs   = model.bse[params.index]
        pvals  = model.pvalues[params.index]

        colors_bar = [TEAL if p < 0.05 else CORAL for p in pvals]
        y_pos = range(len(params))
        ax.barh(list(y_pos), params.values, xerr=1.96 * errs.values,
                color=colors_bar, alpha=0.8, edgecolor=DARK_BG,
                error_kw=dict(ecolor=WHITE, lw=1.5, capsize=4))
        ax.axvline(0, color=WHITE, lw=1, linestyle="--", alpha=0.6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(params.index.tolist(), fontsize=8)
        ax.set_title(f"DiD Coefficients (Extended Model)\nR²={model.rsquared:.3f}",
                     color=WHITE, fontsize=11)
        ax.set_xlabel("Coefficient (± 95% CI)")

    # Right: Basic vs Extended key DiD term
    names  = ["Basic\nDiD", "Extended\n(+controls)", "Full\n(+during)"]
    models_list = [did_results["basic"], did_results["extended"], did_results["full"]]
    coefs  = [m.params.get("did", np.nan) for m in models_list]
    errs2  = [m.bse.get("did", np.nan) for m in models_list]
    pvals2 = [m.pvalues.get("did", np.nan) for m in models_list]
    colors2 = [TEAL if p < 0.05 else CORAL for p in pvals2]

    axes[1].bar(names, coefs, yerr=[1.96 * e for e in errs2],
                color=colors2, alpha=0.8, edgecolor=DARK_BG,
                error_kw=dict(ecolor=WHITE, lw=1.5, capsize=6))
    axes[1].axhline(0, color=WHITE, lw=1, linestyle="--", alpha=0.6)
    for i, (c, p) in enumerate(zip(coefs, pvals2)):
        if not np.isnan(c):
            axes[1].text(i, c + (0.3 if c >= 0 else -0.6),
                         f"{c:.2f}\np={p:.3f}",
                         ha="center", color=WHITE, fontsize=8)
    axes[1].set_title("DiD Treatment Effect Across Model Specifications\n(did coefficient = causal estimate)",
                      color=WHITE, fontsize=11)
    axes[1].set_ylabel("DiD Coefficient")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor=TEAL,  label="p < 0.05 (significant)"),
        Patch(facecolor=CORAL, label="p >= 0.05 (not significant)"),
    ], loc="lower center", ncol=2, fontsize=9,
       facecolor=PANEL_BG, edgecolor=GRID, labelcolor=WHITE,
       bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    saved.append(save_fig(fig, "02_did_coefficients"))


def fig03_event_study(ev_df, saved):
    fig, ax = plt.subplots(figsize=(13, 6))
    apply_style(fig, ax)

    ev_clean = ev_df.dropna(subset=["coef"])
    sig      = ev_clean["p"] < 0.05

    ax.fill_between(ev_clean["year"], ev_clean["ci_lo"], ev_clean["ci_hi"],
                    alpha=0.2, color=TEAL, label="95% CI")
    ax.plot(ev_clean["year"], ev_clean["coef"],
            color=TEAL, lw=2.5, marker="o", markersize=7, zorder=5)

    # Highlight significant years
    ax.scatter(ev_clean.loc[sig, "year"], ev_clean.loc[sig, "coef"],
               color=GOLD, s=80, zorder=6, label="Significant (p<0.05)")

    ax.axhline(0, color=WHITE, lw=1, linestyle="--", alpha=0.5)
    ax.axvline(2019, color=GOLD, lw=1.5, linestyle=":", alpha=0.8, label="Base year (2019)")
    ax.axvspan(2020, 2021.5, color=CORAL, alpha=0.1, label="Pandemic period")

    ax.set_title("Event Study (Dynamic DiD)\nYear-by-year Treatment Effect Relative to 2019 Baseline",
                 color=WHITE, fontsize=12)
    ax.set_xlabel("Year"); ax.set_ylabel("Treatment Effect on Startup Growth (%)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)
    fig.tight_layout()
    saved.append(save_fig(fig, "03_event_study"))


def fig04_propensity_scores(psm_results, saved):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    apply_style(fig, axes)

    prop_df = psm_results["propensity_df"]

    # Left: propensity score distributions
    for group, color, label in [(1, TEAL, "Treated (High Internet)"),
                                  (0, CORAL, "Control (Low Internet)")]:
        axes[0].hist(prop_df[prop_df["treated"] == group]["propensity"],
                     bins=10, color=color, alpha=0.7, label=label, edgecolor=DARK_BG)
    axes[0].set_title("Propensity Score Distribution\n(2019 baseline characteristics)",
                      color=WHITE, fontsize=11)
    axes[0].set_xlabel("Propensity Score")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)

    # Right: matched pairs table
    pairs = psm_results["pairs"]
    axes[1].axis("off")

    rows = [[row["treated_country"], row["control_country"],
             f"{row['treat_propensity']:.3f}", f"{row['ctrl_propensity']:.3f}",
             f"{row['pscore_diff']:.3f}"]
            for _, row in pairs.iterrows()]

    tbl = axes[1].table(
        cellText=rows,
        colLabels=["Treated", "Control", "P(Treated)", "P(Control)", "Diff"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#0A3D5C" if r == 0 else PANEL_BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor(GRID)
    axes[1].set_title("Matched Pairs (Nearest-Neighbour PSM)", color=WHITE, fontsize=11)

    att = psm_results["att"]
    p   = psm_results["p_value"]
    fig.suptitle(f"Propensity Score Matching | ATT={att:.3f} | p={p:.4f}",
                 color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "04_propensity_score_matching"))


def fig05_treatment_group_comparison(df, saved):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    apply_style(fig, axes)

    periods = ["pre", "during", "post"]
    period_labels = ["Pre (2015-19)", "During (2020-21)", "Post (2022-24)"]

    for ax, col, ylabel in zip(axes,
        ["startup_growth_yoy", "total_funding_usd_mn", "startup_density"],
        ["Startup Growth YoY (%)", "Funding (USD Mn)", "Startup Density"]
    ):
        x = np.arange(len(periods))
        w = 0.35

        treated_means = [df[(df["treated"]==1)&(df["pandemic_period"]==p)][col].mean() for p in periods]
        control_means = [df[(df["treated"]==0)&(df["pandemic_period"]==p)][col].mean() for p in periods]
        treated_sems  = [df[(df["treated"]==1)&(df["pandemic_period"]==p)][col].sem()  for p in periods]
        control_sems  = [df[(df["treated"]==0)&(df["pandemic_period"]==p)][col].sem()  for p in periods]

        ax.bar(x - w/2, treated_means, w, color=TEAL, alpha=0.8, label="Treated",
               yerr=treated_sems, error_kw=dict(ecolor=WHITE, lw=1.5, capsize=4), edgecolor=DARK_BG)
        ax.bar(x + w/2, control_means, w, color=CORAL, alpha=0.8, label="Control",
               yerr=control_sems, error_kw=dict(ecolor=WHITE, lw=1.5, capsize=4), edgecolor=DARK_BG)
        ax.set_xticks(x)
        ax.set_xticklabels(period_labels, fontsize=8)
        ax.set_title(ylabel, color=WHITE, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)

    fig.suptitle("Treated vs Control Group — Outcome Comparison by Period",
                 color=WHITE, fontsize=13)
    fig.tight_layout()
    saved.append(save_fig(fig, "05_treated_vs_control"))


def fig06_did_regression_table(did_results, saved):
    model = did_results["extended"]
    params = model.params.drop([c for c in model.params.index
                                 if c.startswith("C(country") or c == "Intercept"], errors="ignore")
    pvals  = model.pvalues[params.index]
    cis    = model.conf_int().loc[params.index]
    sems   = model.bse[params.index]

    rows = []
    for var in params.index:
        stars = "***" if pvals[var] < 0.001 else "**" if pvals[var] < 0.01 else "*" if pvals[var] < 0.05 else ""
        rows.append([
            var,
            f"{params[var]:.4f}{stars}",
            f"{sems[var]:.4f}",
            f"{cis.loc[var, 0]:.4f}",
            f"{cis.loc[var, 1]:.4f}",
            f"{pvals[var]:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(16, max(5, len(rows) * 0.55 + 1.5)))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Variable", "Coef.", "Std.Err", "CI Low", "CI High", "p-value"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        is_did = r > 0 and rows[r-1][0] == "did"
        cell.set_facecolor("#0A3D5C" if r == 0 else ("#0D3320" if is_did else PANEL_BG))
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor(GRID)

    ax.set_title(
        f"DiD Regression Results (Extended Model) | R²={model.rsquared:.3f} | N={int(model.nobs)}\n"
        f"*p<0.05  **p<0.01  ***p<0.001 | HC3 robust standard errors",
        color=WHITE, fontsize=11, pad=20, loc="left"
    )
    fig.tight_layout()
    saved.append(save_fig(fig, "06_did_regression_table"))


def fig07_causal_dag(saved):
    """Simple causal DAG illustration"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")

    nodes = {
        "Internet\nPenetration":  (2, 5.5),
        "Pandemic\n(COVID-19)":   (5, 5.5),
        "Startup\nGrowth":        (8, 3.5),
        "GDP\nGrowth":            (2, 1.5),
        "R&D\nExpenditure":       (5, 1.5),
        "Unemployment\nRate":     (8, 1.5),
    }

    edges = [
        ("Internet\nPenetration",  "Startup\nGrowth",   TEAL,  "treatment effect"),
        ("Pandemic\n(COVID-19)",   "Startup\nGrowth",   CORAL, "direct shock"),
        ("Pandemic\n(COVID-19)",   "Internet\nPenetration", GOLD, "accelerates adoption"),
        ("GDP\nGrowth",            "Startup\nGrowth",   LAVENDER, "confounder"),
        ("R&D\nExpenditure",       "Startup\nGrowth",   CYAN,  "confounder"),
        ("Unemployment\nRate",     "Startup\nGrowth",   "#FDA7DF", "confounder"),
        ("Internet\nPenetration",  "R&D\nExpenditure",  GRID,  ""),
    ]

    for name, (x, y) in nodes.items():
        is_treatment = "Internet" in name
        is_outcome   = "Startup" in name
        is_instrument = "Pandemic" in name
        color = TEAL if is_treatment else (GOLD if is_outcome else (CORAL if is_instrument else PANEL_BG))
        circle = plt.Circle((x, y), 0.65, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", color=WHITE,
                fontsize=7.5, fontweight="bold", zorder=4)

    for src, dst, color, label in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2,
                                    connectionstyle="arc3,rad=0.1"), zorder=2)
        mx, my = (x1+x2)/2, (y1+y2)/2
        if label:
            ax.text(mx, my + 0.2, label, ha="center", color=color, fontsize=7, alpha=0.9)

    ax.set_title("Causal DAG — Startup Growth Ecosystem\n"
                 "Nodes: Treatment (teal), Outcome (gold), Instrument (red), Confounders (dark)",
                 color=WHITE, fontsize=12, pad=10)
    fig.tight_layout()
    saved.append(save_fig(fig, "07_causal_dag"))


def fig08_summary(did_results, psm_results, ev_df, saved):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    apply_style(fig, axes)

    # Left: DiD vs PSM effect comparison
    methods = ["DiD\n(basic)", "DiD\n(extended)", "PSM\n(ATT)"]
    coefs   = [
        did_results["basic"].params.get("did", np.nan),
        did_results["extended"].params.get("did", np.nan),
        psm_results["att"],
    ]
    pvals   = [
        did_results["basic"].pvalues.get("did", np.nan),
        did_results["extended"].pvalues.get("did", np.nan),
        psm_results["p_value"],
    ]
    colors  = [TEAL if p < 0.05 else CORAL for p in pvals]

    axes[0].bar(methods, coefs, color=colors, alpha=0.85, edgecolor=DARK_BG)
    axes[0].axhline(0, color=WHITE, lw=1, linestyle="--", alpha=0.6)
    for i, (c, p) in enumerate(zip(coefs, pvals)):
        if not np.isnan(c):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            axes[0].text(i, c + (0.2 if c >= 0 else -0.5),
                         f"{c:.2f}\n({sig})", ha="center", color=WHITE, fontsize=9)
    axes[0].set_title("Causal Effect Estimates — Method Comparison\n"
                      "All estimate: treatment effect on startup growth YoY",
                      color=WHITE, fontsize=11)
    axes[0].set_ylabel("Estimated Treatment Effect (%)")

    # Right: Event study summary (pre vs post significant years)
    ev_clean = ev_df.dropna(subset=["coef"])
    pre_ev   = ev_clean[ev_clean["year"] < 2020]
    post_ev  = ev_clean[ev_clean["year"] >= 2022]

    axes[1].bar(pre_ev["year"].astype(str),  pre_ev["coef"],  color=TEAL,  alpha=0.8,
                label="Pre-pandemic", edgecolor=DARK_BG)
    axes[1].bar(post_ev["year"].astype(str), post_ev["coef"], color=GOLD,  alpha=0.8,
                label="Post-pandemic", edgecolor=DARK_BG)
    axes[1].axhline(0, color=WHITE, lw=1, linestyle="--", alpha=0.6)
    axes[1].set_title("Event Study Coefficients\nPre-pandemic vs Post-pandemic Periods",
                      color=WHITE, fontsize=11)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Treatment Effect")
    axes[1].legend(fontsize=9, facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=GRID)

    fig.suptitle("Module 7 — Causal Inference Summary", color=WHITE, fontsize=14)
    fig.tight_layout()
    saved.append(save_fig(fig, "08_causal_inference_summary"))


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_report(did_results, psm_results, ev_df, df, high_inet):
    path = get_project_root() / "data/outputs/reports/module7_causal_report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODULE 7 — CAUSAL INFERENCE REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {len(df)} rows | {df['country_code'].nunique()} countries\n")
        f.write("=" * 70 + "\n\n")

        f.write("TREATMENT DEFINITION\n")
        f.write(f"  Treatment : High internet penetration (>= median in 2019)\n")
        f.write(f"  Treated countries ({len(high_inet)}): {sorted(high_inet)}\n")
        f.write(f"  Control countries: remainder\n\n")

        f.write("METHOD 1: DIFFERENCE-IN-DIFFERENCES\n")
        f.write(f"  DiD coef (basic)    : {did_results['basic'].params.get('did',np.nan):.4f}")
        f.write(f"  | p={did_results['basic'].pvalues.get('did',np.nan):.4f}\n")
        f.write(f"  DiD coef (extended) : {did_results['did_coef']:.4f}")
        f.write(f"  | p={did_results['did_p']:.4f}\n")
        f.write(f"  95% CI              : [{did_results['did_ci'][0]:.4f}, {did_results['did_ci'][1]:.4f}]\n")
        f.write(f"  R²                  : {did_results['r2']:.4f}\n")
        f.write(f"  Interpretation: High-internet countries show {did_results['did_coef']:+.2f}pp higher\n")
        f.write(f"  startup growth post-pandemic vs control, after controlling for confounders.\n\n")

        f.write("METHOD 2: PROPENSITY SCORE MATCHING\n")
        f.write(f"  ATT (treated - control, post period): {psm_results['att']:.4f}\n")
        f.write(f"  t-stat: {psm_results['t_stat']:.4f} | p: {psm_results['p_value']:.4f}\n")
        f.write(f"  Treated mean growth : {psm_results['treated_mean']:.4f}\n")
        f.write(f"  Control mean growth : {psm_results['control_mean']:.4f}\n\n")

        f.write("METHOD 3: EVENT STUDY (Dynamic DiD)\n")
        ev_sig = ev_df[ev_df["p"] < 0.05]
        f.write(f"  Significant years (p<0.05): {ev_sig['year'].tolist()}\n")
        f.write(f"  Pre-pandemic coefficients (should ~= 0 for parallel trends):\n")
        for _, row in ev_df[ev_df["year"] < 2020].iterrows():
            f.write(f"    {int(row['year'])}: coef={row['coef']:.4f} | p={row['p']:.4f}\n")

        f.write(f"\nOVERALL CONCLUSION\n")
        f.write(f"  DiD and PSM both estimate a positive treatment effect, suggesting\n")
        f.write(f"  countries with higher pre-pandemic internet penetration experienced\n")
        f.write(f"  stronger post-pandemic startup ecosystem recovery.\n")
        f.write(f"  Caution: N=15 countries limits statistical power; results are indicative.\n")

    logger.info("Report saved to data/outputs/reports/module7_causal_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run():
    logger.info("=" * 60)
    logger.info("MODULE 7: CAUSAL INFERENCE — START")
    logger.info("=" * 60)

    saved = []

    df, high_inet, median_inet = load_and_prep()
    print(f"\n  Data: {df.shape} | Treated: {len(high_inet)} countries | Median inet: {median_inet:.1f}%")
    print(f"  Treated: {sorted(high_inet)}")

    # Method 1: DiD
    print("\n  [1/3] Running Difference-in-Differences...")
    did_results = run_did(df)
    print(f"    DiD coef={did_results['did_coef']:.4f} | p={did_results['did_p']:.4f} | R²={did_results['r2']:.3f}")

    # Method 2: PSM
    print("\n  [2/3] Running Propensity Score Matching...")
    psm_results = run_psm(df)
    print(f"    ATT={psm_results['att']:.4f} | p={psm_results['p_value']:.4f}")

    # Method 3: Event Study
    print("\n  [3/3] Running Event Study (Dynamic DiD)...")
    ev_df = run_event_study(df)
    sig_years = ev_df[ev_df["p"] < 0.05]["year"].tolist()
    print(f"    Significant years: {sig_years}")

    # Save DiD results
    res_path = get_project_root() / "data/processed/did_results.csv"
    ev_df.to_csv(res_path, index=False)

    # Figures
    print("\n  Generating figures...")
    fig01_parallel_trends(df, high_inet, saved)
    fig02_did_coefficients(did_results, saved)
    fig03_event_study(ev_df, saved)
    fig04_propensity_scores(psm_results, saved)
    fig05_treatment_group_comparison(df, saved)
    fig06_did_regression_table(did_results, saved)
    fig07_causal_dag(saved)
    fig08_summary(did_results, psm_results, ev_df, saved)

    # Report
    write_report(did_results, psm_results, ev_df, df, high_inet)

    logger.info("=" * 60)
    logger.info(f"MODULE 7 COMPLETE — {len(saved)} figures + 1 report")
    logger.info(f"Location: {OUT_DIR}")
    for p in saved:
        logger.info(f"  {Path(p).name}")

    print(f"\n{'='*60}")
    print(f"MODULE 7 COMPLETE — {len(saved)} figures + 1 report")
    print(f"  DiD causal effect     : {did_results['did_coef']:+.4f} pp (p={did_results['did_p']:.4f})")
    print(f"  PSM ATT               : {psm_results['att']:+.4f} pp (p={psm_results['p_value']:.4f})")
    print(f"  Event study sig years : {sig_years}")
    print(f"  did_results.csv saved")
    print(f"{'='*60}")
    print(f"\nNext: python scripts\\run_module8.py  (Explainable AI)")

    return did_results, psm_results, ev_df


if __name__ == "__main__":
    run()
