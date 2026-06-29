"""
module8_xai.py — Explainable AI (SHAP + Permutation + PDP)
=============================================================

OBJECTIVE
---------
Explain WHY the best ML model makes its predictions using
three complementary XAI techniques:

1. SHAP (SHapley Additive exPlanations)
   - Global: which features matter most overall
   - Local:  why did the model predict X for country Y

2. Permutation Importance
   - Randomly shuffle one feature at a time
   - Measure how much model R² drops
   - Drop = importance (model-agnostic)

3. Partial Dependence Plots (PDP)
   - Show the marginal effect of one feature on the prediction
   - Holds all other features at their mean

MATHEMATICAL NOTES
------------------
SHAP value for feature i:
  phi_i = sum over all subsets S not containing i:
          [|S|!(p-|S|-1)!/p!] * [f(S + i) - f(S)]
  Interpretation: how much does feature i push the prediction
  above or below the baseline (average prediction)?

Permutation Importance:
  PI_i = score(model, X, y) - score(model, X_shuffled_i, y)
  Positive PI = feature contributes to model accuracy.

Partial Dependence:
  PD(x_s) = E_X_c[f(x_s, X_c)]
  Average prediction when feature x_s varies, others are random.

FIGURES (10)
------------
 1. SHAP summary beeswarm — feature impact distribution
 2. SHAP bar — mean absolute SHAP per feature
 3. SHAP waterfall — single highest-growth country
 4. SHAP waterfall — single lowest-growth country
 5. SHAP dependence — top feature vs SHAP value
 6. SHAP dependence — 2nd feature vs SHAP value
 7. Permutation importance bar chart
 8. PDP — top 2 features (side by side)
 9. PDP — top 3rd and 4th features
10. SHAP interaction heatmap (mean |SHAP| matrix)

INPUTS
------
- data/master_features.csv  (or master_clean.csv as fallback)
- Best ML model from Module 5 (retrained here if pkl not found)

OUTPUTS
-------
- data/outputs/figures/module8/*.png  (10 figures)
- data/outputs/reports/module8_xai_report.txt
"""

import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score
import shap

warnings.filterwarnings("ignore")

# ── Path setup (matches older session convention) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils import get_logger, get_project_root

logger   = get_logger("module8_xai")
ROOT     = get_project_root()

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR  = ROOT / "data" / "outputs" / "figures" / "module8"
REP_DIR  = ROOT / "data" / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
TARGET = "startup_count_yoy"   # will auto-detect if different


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load master_features.csv if it exists, else master_clean.csv.
    Auto-detect target column name.
    """
    candidates = [
        ROOT / "data" / "master_features.csv",
        ROOT / "data" / "processed" / "master_features.csv",
        ROOT / "data" / "master_clean.csv",
        ROOT / "data" / "interim" / "master_raw.csv",
    ]
    df = None
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded: {path.name}  shape={df.shape}")
            break

    if df is None:
        raise FileNotFoundError(
            "No master dataset found. Run Module 6 first to generate master_features.csv"
        )

    # Auto-detect target column
    global TARGET
    target_candidates = [
    "startup_count_yoy", "startup_count_growth_rate",
    "startup_growth_yoy", "yoy_growth", "startup_growth",
    "growth_rate"
]
    for t in target_candidates:
        if t in df.columns:
            TARGET = t
            logger.info(f"Target column: {TARGET}")
            break
    else:
        # Use first numeric column that looks like a rate
        num_cols = df.select_dtypes(include="number").columns
        rate_cols = [c for c in num_cols if "growth" in c or "rate" in c or "yoy" in c]
        if rate_cols:
            TARGET = rate_cols[0]
            logger.info(f"Target auto-detected: {TARGET}")
        else:
            raise ValueError(f"Cannot find target column. Columns: {list(df.columns)}")

    return df


def prepare_features(df):
    """
    Select numeric feature columns, drop target and ID columns.
    Returns X (array), y (array), feature_names (list).
    """
    exclude = {TARGET, "country", "country_code", "year",
               "period", "period_enc", "id"}
    # Drop scaled columns to avoid redundancy
    exclude.update([c for c in df.columns if c.endswith("_scaled")])

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude
    ]

    sub = df[feature_cols + [TARGET]].dropna()
    X   = sub[feature_cols].values
    y   = sub[TARGET].values

    logger.info(f"Features: {len(feature_cols)}  Samples: {len(y)}")
    logger.info(f"Feature list: {feature_cols}")
    return X, y, feature_cols, sub


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def get_model(X, y):
    """
    Load saved model if available, otherwise train Random Forest.
    RF is preferred for SHAP TreeExplainer compatibility.
    """
    model_candidates = [
        ROOT / "models" / "best_model.pkl",
        ROOT / "models" / "saved" / "best_model.pkl",
        ROOT / "data" / "models" / "best_model.pkl",
    ]
    for path in model_candidates:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                model = payload.get("model", payload)
                model.fit(X, y)   # refit on current feature set
                logger.info(f"Loaded model from: {path}")
                return model
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")

    # Train fresh Random Forest
    logger.info("Training fresh Random Forest for XAI...")
    model = RandomForestRegressor(
        n_estimators=300, max_depth=6,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    logger.info(f"RF trained: Train R² = {r2:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X, feature_names):
    """Compute SHAP values using TreeExplainer."""
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    baseline = float(expected_value) if hasattr(expected_value, '__len__') is False else float(expected_value[0])
    logger.info(f"SHAP computed: shape={shap_values.shape}  baseline={baseline:.3f}")
    return shap_values, expected_value, explainer


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {name}")
    return path


def fig1_shap_beeswarm(shap_values, X, feature_names):
    """Fig 1: SHAP beeswarm — shows distribution of impact per feature."""
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, plot_size=None, max_display=15)
    plt.title("SHAP Summary — Feature Impact on Startup Growth Prediction",
              fontsize=13, fontweight="bold", pad=12)
    savefig("01_shap_beeswarm.png")


def fig2_shap_bar(shap_values, feature_names):
    """Fig 2: Mean |SHAP| bar chart."""
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(mean_shap)))
    mean_shap.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("SHAP Feature Importance — Mean |SHAP Value|",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| (average impact on prediction)")
    plt.tight_layout()
    savefig("02_shap_bar.png")


def fig3_shap_waterfall_high(shap_values, expected_value,
                              feature_names, sub_df, explainer):
    """Fig 3: Waterfall for the highest-growth observation."""
    y_vals = sub_df[TARGET].values
    idx    = int(np.argmax(y_vals))
    row    = sub_df.iloc[idx]
    country = row.get("country", f"Row {idx}")

    fig, ax = plt.subplots(figsize=(12, 6))
    shap_series = pd.Series(shap_values[idx], index=feature_names)
    top = shap_series.abs().sort_values(ascending=False).head(10).index
    shap_top = shap_series[top].sort_values()

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in shap_top.values]
    ax.barh(shap_top.index, shap_top.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP Waterfall — Highest Growth: {country}\n"
             f"Actual growth = {float(y_vals[idx]):.2f}%  |  Baseline = {float(np.array(expected_value).flat[0]):.2f}%",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("SHAP Value (impact on prediction)")
    plt.tight_layout()
    savefig("03_shap_waterfall_high.png")


def fig4_shap_waterfall_low(shap_values, expected_value,
                             feature_names, sub_df, explainer):
    """Fig 4: Waterfall for the lowest-growth observation."""
    y_vals  = sub_df[TARGET].values
    idx     = int(np.argmin(y_vals))
    row     = sub_df.iloc[idx]
    country = row.get("country", f"Row {idx}")

    fig, ax = plt.subplots(figsize=(12, 6))
    shap_series = pd.Series(shap_values[idx], index=feature_names)
    top = shap_series.abs().sort_values(ascending=False).head(10).index
    shap_top = shap_series[top].sort_values()

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in shap_top.values]
    ax.barh(shap_top.index, shap_top.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP Waterfall — Highest Growth: {country}\n"
             f"Actual growth = {float(y_vals[idx]):.2f}%  |  Baseline = {float(np.array(expected_value).flat[0]):.2f}%",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("SHAP Value (impact on prediction)")
    plt.tight_layout()
    savefig("04_shap_waterfall_low.png")


def fig5_shap_dependence_top1(shap_values, X, feature_names):
    """Fig 5: SHAP dependence plot — top feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = int(np.argmax(mean_abs))
    top_feat = feature_names[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X[:, top_idx], shap_values[:, top_idx],
               c=shap_values[:, top_idx], cmap="RdYlGn",
               alpha=0.7, s=50, edgecolors="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(top_feat.replace("_", " "))
    ax.set_ylabel(f"SHAP value for {top_feat.replace('_',' ')}")
    ax.set_title(f"SHAP Dependence — {top_feat.replace('_',' ')}",
                 fontsize=12, fontweight="bold")
    plt.colorbar(ax.collections[0], ax=ax, label="SHAP value")
    plt.tight_layout()
    savefig("05_shap_dependence_top1.png")


def fig6_shap_dependence_top2(shap_values, X, feature_names):
    """Fig 6: SHAP dependence plot — 2nd feature."""
    mean_abs  = np.abs(shap_values).mean(axis=0)
    sorted_ix = np.argsort(mean_abs)[::-1]
    idx2      = int(sorted_ix[1]) if len(sorted_ix) > 1 else 0
    feat2     = feature_names[idx2]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X[:, idx2], shap_values[:, idx2],
               c=shap_values[:, idx2], cmap="coolwarm",
               alpha=0.7, s=50, edgecolors="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feat2.replace("_", " "))
    ax.set_ylabel(f"SHAP value for {feat2.replace('_',' ')}")
    ax.set_title(f"SHAP Dependence — {feat2.replace('_',' ')}",
                 fontsize=12, fontweight="bold")
    plt.colorbar(ax.collections[0], ax=ax, label="SHAP value")
    plt.tight_layout()
    savefig("06_shap_dependence_top2.png")


def fig7_permutation_importance(model, X, y, feature_names):
    """Fig 7: Permutation importance — model-agnostic, uses R² drop."""
    logger.info("Computing permutation importance...")
    perm = permutation_importance(model, X, y, n_repeats=20,
                                  random_state=42, n_jobs=-1)
    pi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in pi_df["importance"]]
    ax.barh(pi_df["feature"], pi_df["importance"],
            xerr=pi_df["std"], color=colors,
            edgecolor="white", capsize=3,
            error_kw={"linewidth": 1.2})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Permutation Importance (R² drop when feature is shuffled)\n"
                 "Negative = feature hurts model if removed",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean R² decrease (+/- 1 std)")
    plt.tight_layout()
    savefig("07_permutation_importance.png")
    return pi_df


def fig8_pdp_top2(model, X, feature_names):
    """Fig 8: Partial Dependence Plots — top 2 features."""
    mean_abs = np.abs(
        shap.TreeExplainer(model).shap_values(X)
    ).mean(axis=0)
    top2 = np.argsort(mean_abs)[::-1][:2].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, feat_idx in zip(axes, top2):
        feat_name = feature_names[feat_idx]
        x_vals    = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
        X_temp    = np.tile(X.mean(axis=0), (50, 1))
        X_temp[:, feat_idx] = x_vals
        preds = model.predict(X_temp)

        ax.plot(x_vals, preds, color="#2ecc71", linewidth=2.5)
        ax.fill_between(x_vals, preds - preds.std(),
                        preds + preds.std(), alpha=0.2, color="#2ecc71")
        ax.set_xlabel(feat_name.replace("_", " "))
        ax.set_ylabel("Predicted Growth Rate (%)")
        ax.set_title(f"PDP — {feat_name.replace('_',' ')}", fontsize=11, fontweight="bold")

    fig.suptitle("Partial Dependence Plots — Top 2 Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("08_pdp_top2.png")


def fig9_pdp_top3_4(model, X, feature_names):
    """Fig 9: Partial Dependence Plots — 3rd and 4th features."""
    mean_abs = np.abs(
        shap.TreeExplainer(model).shap_values(X)
    ).mean(axis=0)
    top4  = np.argsort(mean_abs)[::-1][:4].tolist()
    feats = top4[2:4]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, feat_idx in zip(axes, feats):
        feat_name = feature_names[feat_idx]
        x_vals    = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
        X_temp    = np.tile(X.mean(axis=0), (50, 1))
        X_temp[:, feat_idx] = x_vals
        preds = model.predict(X_temp)

        ax.plot(x_vals, preds, color="#e74c3c", linewidth=2.5)
        ax.fill_between(x_vals, preds - preds.std(),
                        preds + preds.std(), alpha=0.2, color="#e74c3c")
        ax.set_xlabel(feat_name.replace("_", " "))
        ax.set_ylabel("Predicted Growth Rate (%)")
        ax.set_title(f"PDP — {feat_name.replace('_',' ')}", fontsize=11, fontweight="bold")

    fig.suptitle("Partial Dependence Plots — 3rd and 4th Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("09_pdp_top3_4.png")


def fig10_shap_heatmap(shap_values, feature_names):
    """Fig 10: SHAP interaction heatmap — mean |SHAP| per feature per sample group."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:10]
    top_names = [feature_names[i] for i in top_idx]
    top_shap  = shap_values[:, top_idx]

    # Bin samples into quartiles by mean SHAP
    sample_means = top_shap.mean(axis=1)
    quartile_labels = pd.qcut(sample_means, q=4,
                               labels=["Q1 Low", "Q2", "Q3", "Q4 High"])
    heat_df = pd.DataFrame(top_shap, columns=top_names)
    heat_df["quartile"] = quartile_labels
    heat_pivot = heat_df.groupby("quartile")[top_names].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat_pivot, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("SHAP Value Heatmap — Mean SHAP by Sample Quartile x Top Features",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("Prediction Quartile")
    plt.tight_layout()
    savefig("10_shap_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(shap_values, feature_names, pi_df, model, X, y):
    mean_abs   = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    r2_train   = r2_score(y, model.predict(X))

    lines = [
        "=" * 60,
        "MODULE 8 — EXPLAINABLE AI REPORT",
        "=" * 60,
        "",
        f"Model used       : {type(model).__name__}",
        f"Train R2         : {r2_train:.4f}",
        f"Samples analysed : {X.shape[0]}",
        f"Features         : {X.shape[1]}",
        "",
        "TOP 5 FEATURES BY MEAN |SHAP|",
        "-" * 40,
    ]
    for i, (feat, val) in enumerate(importance.head(5).items(), 1):
        lines.append(f"  {i}. {feat:35s}  mean|SHAP|={val:.4f}")

    lines += [
        "",
        "TOP 5 FEATURES BY PERMUTATION IMPORTANCE",
        "-" * 40,
    ]
    top_perm = pi_df.sort_values("importance", ascending=False).head(5)
    for i, row in enumerate(top_perm.itertuples(), 1):
        lines.append(f"  {i}. {row.feature:35s}  R2 drop={row.importance:.4f}")

    lines += [
        "",
        "KEY FINDINGS",
        "-" * 40,
        f"  Most impactful feature (SHAP): {importance.index[0]}",
        f"  Most impactful feature (Perm): {pi_df.sort_values('importance',ascending=False).iloc[0]['feature']}",
        f"  SHAP baseline (avg prediction): {float(np.array(shap.TreeExplainer(model).expected_value).flat[0]):.3f}%",
        "",
        "Figures saved: 10",
        f"Location: {FIG_DIR}",
        "",
        "=" * 60,
    ]

    report_path = REP_DIR / "module8_xai_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MODULE 8 — EXPLAINABLE AI")
    logger.info("=" * 60)

    # Load data
    df = load_data()
    X, y, feature_names, sub_df = prepare_features(df)

    # Model
    model = get_model(X, y)

    # SHAP
    shap_values, expected_value, explainer = compute_shap(model, X, feature_names)

    # All 10 figures
    fig1_shap_beeswarm(shap_values, X, feature_names)
    fig2_shap_bar(shap_values, feature_names)
    fig3_shap_waterfall_high(shap_values, expected_value, feature_names, sub_df, explainer)
    fig4_shap_waterfall_low(shap_values, expected_value, feature_names, sub_df, explainer)
    fig5_shap_dependence_top1(shap_values, X, feature_names)
    fig6_shap_dependence_top2(shap_values, X, feature_names)
    pi_df = fig7_permutation_importance(model, X, y, feature_names)
    fig8_pdp_top2(model, X, feature_names)
    fig9_pdp_top3_4(model, X, feature_names)
    fig10_shap_heatmap(shap_values, feature_names)

    # Report
    report_lines = write_report(shap_values, feature_names, pi_df, model, X, y)

    logger.info("=" * 60)
    logger.info("MODULE 8 COMPLETE — 10 figures + 1 report")
    logger.info(f"Location: {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        logger.info(f"  {f.name}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  MODULE 8 COMPLETE")
    print("=" * 60)
    for line in report_lines:
        print(" ", line)
    print("=" * 60)
    print(f"\n  Next: python scripts\\run_module9.py  (Clustering)")


if __name__ == "__main__":
    main()
