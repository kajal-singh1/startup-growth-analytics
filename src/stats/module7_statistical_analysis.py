"""
module7_statistical_analysis.py — Statistical Analysis
========================================================

OBJECTIVE
---------
Apply formal statistical methods to quantify relationships between
features and the target variable (startup_count_growth_rate).
Move from visual patterns (EDA) to numerical evidence with
confidence intervals and p-values.

WHY STATISTICAL ANALYSIS
--------------------------
EDA shows WHAT the data looks like. Statistical analysis answers:
  - Are the correlations we see real or due to chance?
  - Which features significantly predict startup growth?
  - Is the pandemic effect statistically significant when we
    control for other variables?
  - Do different country groups behave differently?

METHODS USED
------------
1. Pearson Correlation   — linear relationship strength
2. Spearman Correlation  — monotonic relationship (rank-based,
                           robust to outliers and non-normality)
3. Multiple Linear Regression — how features jointly predict target
4. Hypothesis Tests:
   a. T-test: pre vs post pandemic growth rates
   b. ANOVA: growth rate differences across country groups
   c. Shapiro-Wilk: normality of target variable
   d. Levene: equality of variances between groups

MATHEMATICAL NOTES
------------------
Pearson r:
  r = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / √[Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²]
  Range: [−1, +1]. |r| > 0.3 = moderate, |r| > 0.5 = strong.

Spearman ρ:
  ρ = 1 − 6Σdᵢ² / n(n²−1)  where dᵢ = rank difference
  Non-parametric — doesn't assume normality.

OLS Regression:
  ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
  Estimated by minimising Σ(yᵢ − ŷᵢ)²
  R² measures proportion of variance explained.

T-test (independent samples):
  t = (ȳ₁ − ȳ₂) / √(s₁²/n₁ + s₂²/n₂)
  H₀: μ₁ = μ₂.  Reject if p < 0.05.

ANOVA (one-way):
  F = variance_between_groups / variance_within_groups
  H₀: all group means equal.  Reject if p < 0.05.

FIGURES (10)
------------
 1. Pearson correlation bar chart — features vs target
 2. Spearman correlation bar chart — features vs target
 3. Pearson vs Spearman comparison
 4. Regression coefficient plot (standardised betas)
 5. Actual vs predicted values
 6. Residuals vs fitted values
 7. Residual distribution (normality check)
 8. Q-Q plot of residuals
 9. Group means with confidence intervals (ANOVA)
10. Feature significance summary (p-value heatmap)

INPUTS
------
- data/processed/master_features.csv

OUTPUTS
-------
- outputs/figures/module7_01_*.png … module7_10_*.png
- outputs/reports/module7_stats_report.txt
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (load_config, setup_logging, save_figure,
                   load_dataframe, write_module_summary, set_seeds)

sns.set_theme(style="whitegrid", palette="muted")
TARGET = "startup_count_growth_rate"

FEATURE_COLS = [
    "gdp_growth_rate",
    "internet_penetration_pct",
    "gdp_per_capita_usd",
    "unemployment_rate",
    "innovation_score",
    "digital_readiness_score",
    "economic_momentum",
    "investment_efficiency",
    "startup_density",
    "pandemic_period",
    "pandemic_interaction",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(df, logger):
    """
    Compute Pearson and Spearman correlations of each feature with target.

    Returns
    -------
    pd.DataFrame with columns: feature, pearson_r, pearson_p,
                                spearman_r, spearman_p
    """
    cols = [c for c in FEATURE_COLS if c in df.columns]
    results = []
    for col in cols:
        sub = df[[col, TARGET]].dropna()
        pr, pp = stats.pearsonr(sub[col], sub[TARGET])
        sr, sp = stats.spearmanr(sub[col], sub[TARGET])
        results.append({
            "feature":    col,
            "pearson_r":  round(pr, 4),
            "pearson_p":  round(pp, 4),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 4),
            "pearson_sig":  "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "",
            "spearman_sig": "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else "",
        })
    corr_df = pd.DataFrame(results).sort_values("pearson_r", key=abs, ascending=False)
    logger.info(f"Correlations computed for {len(corr_df)} features")
    for _, row in corr_df.iterrows():
        logger.info(f"  {row['feature']:35s} pearson={row['pearson_r']:+.3f}{row['pearson_sig']:3s}  "
                    f"spearman={row['spearman_r']:+.3f}{row['spearman_sig']}")
    return corr_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTIPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_regression(df, logger):
    """
    OLS multiple linear regression: target ~ selected features.

    Uses standardised features so coefficients are directly comparable
    (standardised beta = effect of 1 SD change in predictor).

    Returns
    -------
    result : statsmodels RegressionResults
    reg_df : DataFrame of coefficients with confidence intervals
    """
    # Use scaled features where available, else raw
    reg_features = []
    for col in FEATURE_COLS:
        scaled = col + "_scaled"
        if scaled in df.columns:
            reg_features.append(scaled)
        elif col in df.columns:
            reg_features.append(col)

    # Target: use scaled version if available
    target_col = TARGET + "_scaled" if TARGET + "_scaled" in df.columns else TARGET

    sub = df[reg_features + [target_col]].dropna()
    X = sm.add_constant(sub[reg_features])
    y = sub[target_col]

    model  = sm.OLS(y, X)
    result = model.fit()

    logger.info(f"\nOLS Regression Summary:")
    logger.info(f"  R²        = {result.rsquared:.4f}")
    logger.info(f"  Adj. R²   = {result.rsquared_adj:.4f}")
    logger.info(f"  F-stat    = {result.fvalue:.4f}  (p={result.f_pvalue:.4f})")
    logger.info(f"  N         = {int(result.nobs)}")
    logger.info(f"  AIC       = {result.aic:.2f}")

    # Coefficient table
    coef_df = pd.DataFrame({
        "feature":   result.params.index,
        "coef":      result.params.values,
        "std_err":   result.bse.values,
        "t_stat":    result.tvalues.values,
        "p_value":   result.pvalues.values,
        "ci_lower":  result.conf_int()[0].values,
        "ci_upper":  result.conf_int()[1].values,
    })
    coef_df = coef_df[coef_df["feature"] != "const"]
    coef_df["significant"] = coef_df["p_value"] < 0.05
    coef_df = coef_df.sort_values("coef", key=abs, ascending=False)

    for _, row in coef_df.iterrows():
        sig = "*" if row["significant"] else ""
        logger.info(f"  {row['feature']:40s} beta={row['coef']:+.4f}  p={row['p_value']:.4f} {sig}")
    return result, coef_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYPOTHESIS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_hypothesis_tests(df, logger):
    """
    Run all hypothesis tests. Returns dict of results.
    """
    tests = {}

    # ── A. Normality of target (Shapiro-Wilk) ────────────────────────────────
    target = df[TARGET].dropna()
    sw_stat, sw_p = stats.shapiro(target[:50])   # Shapiro-Wilk max n=5000
    tests["shapiro_wilk"] = {
        "statistic": round(sw_stat, 4),
        "p_value":   round(sw_p, 4),
        "normal":    sw_p > 0.05,
        "interpretation": "Target is normally distributed" if sw_p > 0.05
                          else "Target is NOT normally distributed (use Spearman)"
    }
    logger.info(f"Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f} -> {tests['shapiro_wilk']['interpretation']}")

    # ── B. T-test: pre vs post pandemic ──────────────────────────────────────
    pre  = df[df["pandemic_period"] == 0][TARGET].dropna()
    post = df[df["pandemic_period"] == 1][TARGET].dropna()
    lev_stat, lev_p = stats.levene(pre, post)
    equal_var = lev_p > 0.05
    t_stat, t_p = stats.ttest_ind(pre, post, equal_var=equal_var)
    tests["ttest_pandemic"] = {
        "pre_mean":    round(pre.mean(), 3),
        "post_mean":   round(post.mean(), 3),
        "t_statistic": round(t_stat, 4),
        "p_value":     round(t_p, 4),
        "significant": t_p < 0.05,
        "levene_p":    round(lev_p, 4),
        "interpretation": (
            f"Significant difference between pre ({pre.mean():.2f}%) "
            f"and post ({post.mean():.2f}%) pandemic growth (p={t_p:.4f})"
            if t_p < 0.05 else
            f"No significant difference between pre ({pre.mean():.2f}%) "
            f"and post ({post.mean():.2f}%) pandemic growth (p={t_p:.4f})"
        )
    }
    logger.info(f"T-test pandemic: t={t_stat:.4f}, p={t_p:.4f} -> {tests['ttest_pandemic']['interpretation']}")

    # ── C. ANOVA: growth rate by GDP quartile group ───────────────────────────
    df = df.copy()
    df["gdp_group"] = pd.qcut(df["gdp_per_capita_usd"], q=3,
                               labels=["Low GDP", "Mid GDP", "High GDP"])
    groups = [grp[TARGET].dropna().values
              for _, grp in df.groupby("gdp_group", observed=True)]
    f_stat, f_p = stats.f_oneway(*groups)
    tests["anova_gdp_group"] = {
        "f_statistic": round(f_stat, 4),
        "p_value":     round(f_p, 4),
        "significant": f_p < 0.05,
        "group_means": {
            str(name): round(grp[TARGET].mean(), 2)
            for name, grp in df.groupby("gdp_group", observed=True)
        },
        "interpretation": (
            "Significant growth rate differences across GDP groups"
            if f_p < 0.05 else
            "No significant growth rate differences across GDP groups"
        )
    }
    logger.info(f"ANOVA GDP groups: F={f_stat:.4f}, p={f_p:.4f} -> {tests['anova_gdp_group']['interpretation']}")

    # ── D. Correlation significance summary ───────────────────────────────────
    sig_features = []
    for col in FEATURE_COLS:
        if col in df.columns:
            sub = df[[col, TARGET]].dropna()
            _, p = stats.pearsonr(sub[col], sub[TARGET])
            if p < 0.05:
                sig_features.append(col)
    tests["significant_features"] = sig_features
    logger.info(f"Features significantly correlated with target: {sig_features}")

    return tests, df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig1_pearson_bar(corr_df, config, logger):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_df["pearson_r"]]
    bars = ax.barh(corr_df["feature"], corr_df["pearson_r"], color=colors, edgecolor="white")
    for bar, sig in zip(bars, corr_df["pearson_sig"]):
        if sig:
            ax.text(bar.get_width() + 0.01 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height()/2,
                    sig, va="center", fontsize=10, color="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.3,  color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(-0.3, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Pearson Correlation with Startup Growth Rate\n(* p<0.05  ** p<0.01  *** p<0.001)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    save_figure(fig, "module7_01_pearson_correlation.png", config)
    logger.info("Fig 1 saved")


def fig2_spearman_bar(corr_df, config, logger):
    sorted_df = corr_df.sort_values("spearman_r", key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in sorted_df["spearman_r"]]
    bars = ax.barh(sorted_df["feature"], sorted_df["spearman_r"], color=colors, edgecolor="white")
    for bar, sig in zip(bars, sorted_df["spearman_sig"]):
        if sig:
            ax.text(bar.get_width() + 0.01 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height()/2,
                    sig, va="center", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman Correlation with Startup Growth Rate\n(rank-based, robust to outliers)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Spearman ρ")
    plt.tight_layout()
    save_figure(fig, "module7_02_spearman_correlation.png", config)
    logger.info("Fig 2 saved")


def fig3_pearson_vs_spearman(corr_df, config, logger):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(corr_df))
    w = 0.38
    ax.bar(x - w/2, corr_df["pearson_r"],  width=w, label="Pearson r",  color="#2ecc71", alpha=0.85)
    ax.bar(x + w/2, corr_df["spearman_r"], width=w, label="Spearman ρ", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_df["feature"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Pearson vs Spearman Correlation — Feature vs Target", fontsize=12, fontweight="bold")
    ax.set_ylabel("Correlation Coefficient"); ax.legend()
    plt.tight_layout()
    save_figure(fig, "module7_03_pearson_vs_spearman.png", config)
    logger.info("Fig 3 saved")


def fig4_regression_coefs(coef_df, config, logger):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in coef_df["coef"]]
    ax.barh(coef_df["feature"].str.replace("_scaled", "").str.replace("_", " "),
            coef_df["coef"], color=colors, edgecolor="white")
    ax.errorbar(coef_df["coef"],
                range(len(coef_df)),
                xerr=[coef_df["coef"] - coef_df["ci_lower"],
                      coef_df["ci_upper"] - coef_df["coef"]],
                fmt="none", color="black", linewidth=1.2, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("OLS Regression — Standardised Coefficients (with 95% CI)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Standardised β (effect of 1 SD change)")
    plt.tight_layout()
    save_figure(fig, "module7_04_regression_coefficients.png", config)
    logger.info("Fig 4 saved")


def fig5_actual_vs_predicted(df, result, config, logger):
    target_col = TARGET + "_scaled" if TARGET + "_scaled" in df.columns else TARGET
    reg_features = []
    for col in FEATURE_COLS:
        scaled = col + "_scaled"
        if scaled in df.columns:
            reg_features.append(scaled)
        elif col in df.columns:
            reg_features.append(col)
    sub = df[reg_features + [target_col]].dropna()
    X   = sm.add_constant(sub[reg_features])
    y_actual = sub[target_col].values
    y_pred   = result.predict(X).values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_actual, y_pred, alpha=0.6, color="#3498db", edgecolors="white", s=50)
    lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted (R² = {result.rsquared:.3f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Growth Rate (scaled)")
    ax.set_ylabel("Predicted Growth Rate (scaled)")
    ax.legend(); plt.tight_layout()
    save_figure(fig, "module7_05_actual_vs_predicted.png", config)
    logger.info("Fig 5 saved")


def fig6_residuals(result, config, logger):
    fitted    = result.fittedvalues
    residuals = result.resid
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, residuals, alpha=0.6, color="#e67e22", edgecolors="white", s=50)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Residuals vs Fitted Values", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
    plt.tight_layout()
    save_figure(fig, "module7_06_residuals_vs_fitted.png", config)
    logger.info("Fig 6 saved")


def fig7_residual_distribution(result, config, logger):
    residuals = result.resid
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(residuals, bins=25, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[0].set_title("Residual Distribution"); axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    # Overlay normal curve
    x = np.linspace(residuals.min(), residuals.max(), 100)
    mu, sigma = residuals.mean(), residuals.std()
    axes[0].plot(x, len(residuals) * (residuals.max()-residuals.min())/25
                 * stats.norm.pdf(x, mu, sigma), "k--", linewidth=1.5, label="Normal")
    axes[0].legend()
    sm.qqplot(residuals, line="s", ax=axes[1], alpha=0.6)
    axes[1].set_title("Q-Q Plot of Residuals")
    fig.suptitle("Regression Residual Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "module7_07_residual_distribution.png", config)
    logger.info("Fig 7 saved")


def fig8_group_means(df, tests, config, logger):
    df = df.copy()
    df["gdp_group"] = pd.qcut(df["gdp_per_capita_usd"], q=3,
                               labels=["Low GDP", "Mid GDP", "High GDP"])
    group_stats = df.groupby("gdp_group", observed=True)[TARGET].agg(["mean","sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = ax.bar(group_stats["gdp_group"].astype(str),
                  group_stats["mean"], color=colors,
                  yerr=group_stats["sem"] * 1.96,
                  capsize=6, edgecolor="white", alpha=0.85,
                  error_kw={"linewidth": 1.5})
    f_p = tests["anova_gdp_group"]["p_value"]
    sig = "***" if f_p < 0.001 else "**" if f_p < 0.01 else "*" if f_p < 0.05 else "n.s."
    ax.set_title(f"Mean Growth Rate by GDP Group (ANOVA p={f_p:.4f} {sig})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Startup Growth Rate (%)"); ax.set_xlabel("GDP per Capita Group")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    save_figure(fig, "module7_08_group_means_anova.png", config)
    logger.info("Fig 8 saved")


def fig9_pvalue_heatmap(corr_df, config, logger):
    """Significance heatmap: Pearson and Spearman p-values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    heat_data = corr_df.set_index("feature")[["pearson_p", "spearman_p"]]
    heat_data.columns = ["Pearson p", "Spearman p"]
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=0.1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "p-value (green = significant)"})
    ax.set_title("Feature Significance (p-values)\nGreen = p < 0.05 (significant)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "module7_09_pvalue_heatmap.png", config)
    logger.info("Fig 9 saved")


def fig10_correlation_scatter_grid(df, corr_df, config, logger):
    """Scatter plots of top 4 correlated features vs target."""
    top4 = corr_df.head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feat in zip(axes.flat, top4):
        sub = df[[feat, TARGET]].dropna()
        r, p = stats.pearsonr(sub[feat], sub[TARGET])
        ax.scatter(sub[feat], sub[TARGET], alpha=0.6, s=40,
                   color="#3498db", edgecolors="white")
        # Regression line
        slope, intercept = np.polyfit(sub[feat], sub[TARGET], 1)
        x_line = np.linspace(sub[feat].min(), sub[feat].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "r--", linewidth=1.5)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"{feat.replace('_',' ')}\nr={r:.3f} {sig}", fontsize=10)
        ax.set_xlabel(feat.replace("_", " ")); ax.set_ylabel("Growth Rate (%)")
    fig.suptitle("Top 4 Features vs Startup Growth Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "module7_10_top_features_scatter.png", config)
    logger.info("Fig 10 saved")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(config=None):
    if config is None:
        config = load_config()
    set_seeds(config)
    logger = setup_logging("module7_stats", config)

    logger.info("=" * 55)
    logger.info("MODULE 7 — STATISTICAL ANALYSIS")
    logger.info("=" * 55)

    df = load_dataframe("master_features.csv", stage="processed", config=config)
    logger.info(f"Loaded: {df.shape}")

    # Run analyses
    corr_df              = compute_correlations(df, logger)
    result, coef_df      = run_regression(df, logger)
    tests, df            = run_hypothesis_tests(df, logger)

    # Figures
    fig1_pearson_bar(corr_df, config, logger)
    fig2_spearman_bar(corr_df, config, logger)
    fig3_pearson_vs_spearman(corr_df, config, logger)
    fig4_regression_coefs(coef_df, config, logger)
    fig5_actual_vs_predicted(df, result, config, logger)
    fig6_residuals(result, config, logger)
    fig7_residual_distribution(result, config, logger)
    fig8_group_means(df, tests, config, logger)
    fig9_pvalue_heatmap(corr_df, config, logger)
    fig10_correlation_scatter_grid(df, corr_df, config, logger)

    # Significant features (p < 0.05)
    sig_pearson  = corr_df[corr_df["pearson_p"]  < 0.05]["feature"].tolist()
    sig_spearman = corr_df[corr_df["spearman_p"] < 0.05]["feature"].tolist()

    summary = {
        "Dataset shape":                  str(df.shape),
        "Features tested":                len(corr_df),
        "Sig. Pearson features (p<0.05)": sig_pearson or "None",
        "Sig. Spearman features (p<0.05)":sig_spearman or "None",
        "OLS R²":                         round(result.rsquared, 4),
        "OLS Adjusted R²":                round(result.rsquared_adj, 4),
        "OLS F-statistic p-value":        round(result.f_pvalue, 4),
        "Shapiro-Wilk (normality)":       tests["shapiro_wilk"]["interpretation"],
        "T-test pandemic effect":         tests["ttest_pandemic"]["interpretation"],
        "ANOVA GDP groups":               tests["anova_gdp_group"]["interpretation"],
        "Top feature by Pearson":         corr_df.iloc[0]["feature"],
        "Top Pearson r":                  corr_df.iloc[0]["pearson_r"],
        "Figures saved":                  10,
        "Status":                         "COMPLETE",
    }

    write_module_summary("module7_stats", summary, config)

    print("\n" + "="*55)
    print("  MODULE 7 — STATISTICAL ANALYSIS REPORT")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  ✓ 10 figures saved to outputs/figures/")
    print(f"  ✓ Report saved to outputs/reports/")
    print("="*55)

    return df, corr_df, result, tests


if __name__ == "__main__":
    run()
