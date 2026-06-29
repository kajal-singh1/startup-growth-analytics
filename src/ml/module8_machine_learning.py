"""
MODULE 8 — MACHINE LEARNING
============================================================================
OBJECTIVE
  Predict year-over-year startup-count growth rate from economic, digital,
  and engineered ecosystem features, and compare a linear baseline against
  three tree-based models to see how much non-linear structure the simpler
  OLS regression in Module 7 was missing.

WHY THIS MODULE IS NEEDED
  Module 7's OLS regression explained only ~24% of the variance in growth
  rate (R^2 = 0.24), and the target failed the Shapiro-Wilk normality test.
  Both findings point the same direction: the relationship between these
  features and startup growth is probably non-linear and/or involves
  interactions that a single linear equation can't capture (e.g. internet
  penetration might only matter when GDP per capita is also high). Tree
  ensembles model exactly that kind of conditional structure natively.

THEORY / MATH INTUITION
  - Linear Regression: y = b0 + b1*x1 + ... + bn*xn + e. Assumes additive,
    constant-slope effects. Our baseline / sanity check.
  - Random Forest: averages many decision trees, each trained on a bootstrap
    resample with a random subset of features per split. Reduces variance
    via averaging (bagging) — robust, but only mitigates, doesn't eliminate,
    overfitting on small data.
  - XGBoost / LightGBM: gradient boosting — trees are built sequentially,
    each new tree fit to the RESIDUAL errors of the ensemble so far. Higher
    capacity to capture subtle patterns, but more prone to overfitting with
    only ~135 rows, which is why cross-validation (not just train/test) is
    used to judge generalization before trusting any single number.

ALGORITHMS USED
  Linear Regression, Random Forest, XGBoost, LightGBM | K-Fold CV | SHAP

INPUTS
  data/processed/master_features.csv  (135 rows x 30 cols from Module 5)

OUTPUTS
  - models/saved/module8_best_model.joblib
  - 10 figures  -> outputs/figures/
  - 2 reports   -> outputs/reports/ (concise summary + full module report)

CONNECTION TO THE NEXT MODULE
  Module 9 (Hyperparameter Tuning) takes whichever model wins here and tunes
  it properly (Grid/Random Search / Optuna) instead of using default params.

Run via: python scripts/run_module8.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (load_config, setup_logging, set_seeds, save_figure,
                    save_dataframe, load_dataframe, get_db_connection,
                    write_module_summary, get_path)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODULE_NAME = "module8_machine_learning"


# --------------------------------------------------------------------------- #
# FEATURE PREPARATION
# --------------------------------------------------------------------------- #
def prepare_features(df, target_col):
    """
    Build the model-ready feature matrix.

    Deliberately EXCLUDED:
      - country / country_code : 15 categories on 135 rows -> one-hot would
        risk severe overfitting on a panel this small.
      - all '*_scaled' columns  : exact duplicates of raw columns already in
        the feature set; including both would just double-count the same
        signal and inflate SHAP/importance scores artificially.
      - target's own scaled twin: that IS the target, just z-scored -> using
        it as a feature would be direct leakage.
    """
    exclude = {target_col, f"{target_col}_scaled", "country", "country_code"}
    scaled_cols = [c for c in df.columns if c.endswith("_scaled")]
    exclude.update(scaled_cols)

    feature_df = df.drop(columns=[c for c in exclude if c in df.columns]).copy()

    # one-hot encode the categorical period indicator (avoids implying a false
    # ordinal relationship between pre/during/post)
    if "pandemic_period" in feature_df.columns:
        feature_df = pd.get_dummies(feature_df, columns=["pandemic_period"],
                                     prefix="period", drop_first=True)

    X = feature_df
    y = df[target_col]
    return X, y


# --------------------------------------------------------------------------- #
# MODEL TRAINING + CROSS-VALIDATION
# --------------------------------------------------------------------------- #
def build_models(config):
    seed = config["reproducibility"]["random_seed"]
    n_jobs = config["ml"]["n_jobs"]
    models = {}

    models["Linear Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=300, max_depth=6, random_state=seed, n_jobs=n_jobs)

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=seed, n_jobs=n_jobs, verbosity=0)
    except ImportError:
        pass  # flagged later via missing_models

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=seed, n_jobs=n_jobs, verbosity=-1)
    except ImportError:
        pass

    expected = set(config["ml"]["models"])
    built_lower = {k.lower().replace(" ", "_") for k in models}
    missing = [m for m in expected if m not in built_lower]
    return models, missing


def evaluate_models(models, X_train, X_test, y_train, y_test, cv_folds, seed, log):
    rows = []
    fitted = {}
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for name, model in models.items():
        log.info(f"Training {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        rows.append({
            "model": name,
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_r2": round(r2, 4),
        })
        fitted[name] = model
        log.info(f"  {name}: CV R2={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
                  f"Test R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

    results_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    return results_df, fitted


# --------------------------------------------------------------------------- #
# SHAP EXPLAINABILITY
# --------------------------------------------------------------------------- #
def compute_shap(fitted, X_train, log):
    """
    SHAP (SHapley Additive exPlanations) attributes each prediction to its
    input features using cooperative-game-theory fair-credit allocation.
    We default to the Random Forest model: it's the most version-robust
    choice for SHAP's TreeExplainer. We still try XGBoost if available,
    but never let a SHAP/XGBoost version mismatch break the whole module.
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed - skipping explainability section. "
                     "Install with: pip install shap")
        return None

    target_model = fitted.get("Random Forest")
    model_used = "Random Forest"
    if target_model is None:
        log.warning("Random Forest unavailable - SHAP section skipped.")
        return None

    try:
        explainer = shap.TreeExplainer(target_model)
        shap_values = explainer(X_train)
    except Exception as e:
        log.warning(f"SHAP failed on {model_used} ({e}) - skipping explainability section.")
        return None

    log.info(f"SHAP values computed using {model_used}")
    return {"explainer": explainer, "shap_values": shap_values, "model_name": model_used, "X": X_train}


# --------------------------------------------------------------------------- #
# FIGURES
# --------------------------------------------------------------------------- #
def fig01_model_comparison(results_df, out_path, config):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results_df))
    ax.bar(x, results_df["cv_r2_mean"], yerr=results_df["cv_r2_std"], capsize=5,
           color="#2563eb", alpha=0.85)
    ax.axhline(config["ml"]["target_r2"], color="#dc2626", ls="--", lw=1.5,
               label=f"Target R² = {config['ml']['target_r2']}")
    ax.set_xticks(x); ax.set_xticklabels(results_df["model"], rotation=15)
    ax.set_ylabel("Cross-validated R²"); ax.set_title("Model Comparison — CV R² (mean ± std)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "module8_01_model_comparison.png", config)


def fig02_predicted_vs_actual(y_test, preds, best_name, out_path, config):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, preds, alpha=0.7, color="#2563eb", edgecolor="white")
    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual growth rate (%)"); ax.set_ylabel("Predicted growth rate (%)")
    ax.set_title(f"Predicted vs Actual — {best_name}")
    ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, "module8_02_predicted_vs_actual.png", config)


def fig03_residual_analysis(y_test, preds, best_name, config):
    residuals = y_test.values - preds
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].scatter(preds, residuals, alpha=0.7, color="#7c3aed")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals vs Fitted")

    axes[1].hist(residuals, bins=15, color="#16a34a", edgecolor="white")
    axes[1].set_title("Residual Distribution"); axes[1].set_xlabel("Residual")

    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")

    fig.suptitle(f"Residual Diagnostics — {best_name}", y=1.02)
    save_figure(fig, "module8_03_residual_analysis.png", config)


def fig04_feature_importance(model, feature_names, title, filename, config):
    if model is None or not hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"{title}: model not available in this environment.",
                 ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, filename, config)
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index[::-1], importances.values[::-1], color="#ea580c")
    ax.set_title(title); ax.set_xlabel("Importance")
    save_figure(fig, filename, config)


def fig06_shap_bar(shap_result, config):
    if shap_result is None:
        _placeholder("module8_06_shap_summary_bar.png", "SHAP not available — see report for install steps.", config)
        return
    import shap
    fig = plt.figure(figsize=(8, 6))
    shap.plots.bar(shap_result["shap_values"], show=False)
    save_figure(plt.gcf(), "module8_06_shap_summary_bar.png", config)


def fig07_shap_beeswarm(shap_result, config):
    if shap_result is None:
        _placeholder("module8_07_shap_beeswarm.png", "SHAP not available — see report for install steps.", config)
        return
    import shap
    fig = plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap_result["shap_values"], show=False)
    save_figure(plt.gcf(), "module8_07_shap_beeswarm.png", config)


def fig08_shap_dependence(shap_result, config):
    if shap_result is None:
        _placeholder("module8_08_shap_dependence.png", "SHAP not available — see report for install steps.", config)
        return
    import shap
    mean_abs = np.abs(shap_result["shap_values"].values).mean(axis=0)
    top_feature = shap_result["X"].columns[np.argmax(mean_abs)]
    fig = plt.figure(figsize=(7, 5))
    shap.plots.scatter(shap_result["shap_values"][:, top_feature], show=False)
    plt.title(f"SHAP Dependence — {top_feature}")
    save_figure(plt.gcf(), "module8_08_shap_dependence.png", config)


def fig09_learning_curves(fitted, X_train, y_train, cv_folds, seed, config):
    candidates = {k: v for k, v in fitted.items() if k in ("Random Forest", "XGBoost", "LightGBM")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"Random Forest": "#2563eb", "XGBoost": "#dc2626", "LightGBM": "#16a34a"}
    for name, model in candidates.items():
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv_folds, scoring="r2",
            train_sizes=np.linspace(0.3, 1.0, 5), random_state=seed)
        ax.plot(sizes, val_scores.mean(axis=1), marker="o", color=colors.get(name, "grey"), label=f"{name} (val)")
        ax.plot(sizes, train_scores.mean(axis=1), marker="o", ls="--", color=colors.get(name, "grey"),
                alpha=0.5, label=f"{name} (train)")
    ax.set_xlabel("Training set size"); ax.set_ylabel("R²")
    ax.set_title("Learning Curves — Bias/Variance Tradeoff")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, "module8_09_learning_curves.png", config)


def fig10_leaderboard(results_df, config):
    fig, ax = plt.subplots(figsize=(9, 1 + 0.6 * len(results_df)))
    ax.axis("off")
    show = results_df.copy()
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center",
                   colWidths=[0.22, 0.16, 0.16, 0.15, 0.15, 0.15])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.8)
    for j in range(len(show.columns)):
        tbl[0, j].set_facecolor("#1f2937"); tbl[0, j].set_text_props(color="white", weight="bold")
    for j in range(len(show.columns)):
        tbl[1, j].set_facecolor("#dcfce7")  # highlight the leader row (already sorted best-first)
    ax.set_title("Model Leaderboard", pad=16)
    save_figure(fig, "module8_10_leaderboard.png", config)


def _placeholder(filename, message, config):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=11, wrap=True)
    ax.axis("off")
    save_figure(fig, filename, config)


# --------------------------------------------------------------------------- #
# FULL MODULE REPORT (per the project spec's documentation requirements)
# --------------------------------------------------------------------------- #
def build_full_report(config, results_df, best_name, missing_models, shap_result, target_col, n_features):
    target_r2 = config["ml"]["target_r2"]
    best_r2 = results_df.iloc[0]["test_r2"]
    lines = []
    lines.append("=" * 72)
    lines.append("MODULE 8 — MACHINE LEARNING — FULL REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    lines.append("\nOBJECTIVE")
    lines.append("-" * 72)
    lines.append(f"Predict '{target_col}' from {n_features} economic/digital/engineered features,")
    lines.append("comparing a linear baseline against tree-ensemble models.")

    lines.append("\nWHY THIS MODULE IS NEEDED")
    lines.append("-" * 72)
    lines.append("Module 7's OLS regression reached R²=0.24 and the target failed normality")
    lines.append("(Shapiro-Wilk p=0.0003) — both symptoms of non-linear structure a single")
    lines.append("linear equation can't capture. Tree ensembles model that structure directly.")

    lines.append("\nALGORITHMS USED")
    lines.append("-" * 72)
    lines.append("Linear Regression (baseline) | Random Forest | XGBoost | LightGBM")
    lines.append("K-Fold cross-validation (k=" + str(config["ml"]["cv_folds"]) + ") | SHAP explainability")
    if missing_models:
        lines.append(f"NOT RUN (not installed in this environment): {', '.join(missing_models)}")
        lines.append("Install with: pip install " + " ".join(missing_models))

    lines.append("\nRESULTS — MODEL LEADERBOARD")
    lines.append("-" * 72)
    lines.append(results_df.to_string(index=False))

    lines.append(f"\nBest model: {best_name}  (Test R² = {best_r2:.4f}, target = {target_r2})")
    if best_r2 >= target_r2:
        lines.append("Target R² ACHIEVED.")
    else:
        lines.append(f"Target R² NOT YET achieved ({best_r2:.2f} < {target_r2}). With only 135 rows of")
        lines.append("panel data and a noisy growth-rate target, this is expected — Module 9")
        lines.append("(Hyperparameter Tuning) and richer features may close some of the gap, but")
        lines.append("a ceiling well below 0.75 on a dataset this size and noisy would not be surprising.")

    lines.append("\nEXPLAINABLE AI (SHAP)")
    lines.append("-" * 72)
    if shap_result is None:
        lines.append("SHAP was not available in this run — see install note above (if applicable),")
        lines.append("or check the install logs for a shap/xgboost version conflict.")
    else:
        mean_abs = np.abs(shap_result["shap_values"].values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:5]
        lines.append(f"Computed using: {shap_result['model_name']}")
        lines.append("Top 5 features by mean |SHAP value|:")
        for i in order:
            lines.append(f"  {shap_result['X'].columns[i]:30s} {mean_abs[i]:.4f}")

    lines.append("\nLIMITATIONS")
    lines.append("-" * 72)
    lines.append("- 135 rows total (15 countries x 9 years) is small for 4 models + CV; results")
    lines.append("  carry real sampling uncertainty (see CV std in the leaderboard above).")
    lines.append("- Train/test split is a random row split, not split by year or by country -")
    lines.append("  rows from the same country appear in both train and test, which can inflate")
    lines.append("  apparent performance vs. a true held-out-country or held-out-year evaluation.")
    lines.append("- Gradient boosting models (XGBoost/LightGBM) are especially prone to overfitting")
    lines.append("  at this sample size; trust the CV mean/std over the single test-set number.")

    lines.append("\nPRACTICAL INTERPRETATION")
    lines.append("-" * 72)
    lines.append("Treat these models as exploratory signal-finders, not production growth")
    lines.append("forecasters yet. The SHAP ranking is more trustworthy for 'which factors matter")
    lines.append("most' than the raw R² is for 'how accurately can we predict next year's growth'.")

    lines.append("\nBEST PRACTICES APPLIED")
    lines.append("-" * 72)
    lines.append("- Cross-validation (not just a single train/test split) used to judge generalization.")
    lines.append("- Scaling only applied to Linear Regression (tree models don't need/want it).")
    lines.append("- Scaled-duplicate and target-leakage columns explicitly excluded from features.")
    lines.append("- Random seed fixed via config for reproducibility.")

    lines.append("\nCONNECTION TO THE NEXT MODULE")
    lines.append("-" * 72)
    lines.append(f"Module 9 (Hyperparameter Tuning) will tune {best_name} specifically — Grid/Random")
    lines.append("Search and Optuna — instead of the default hyperparameters used here.")

    lines.append("\n" + "=" * 72)
    lines.append("MODULE 8 COMPLETE")
    lines.append("=" * 72)

    reports_dir = get_path("reports", config)
    out_path = reports_dir / "module8_machine_learning_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------------------------------- #
def run(config=None):
    if config is None:
        config = load_config()
    set_seeds(config)
    log = setup_logging(MODULE_NAME, config)

    log.info("=" * 60)
    log.info("MODULE 8 — MACHINE LEARNING")
    log.info("=" * 60)

    df = load_dataframe("master_features.csv", stage="processed", config=config)
    target_col = config["data"]["target_variable"]
    log.info(f"Loaded master_features.csv: {df.shape[0]} rows x {df.shape[1]} cols")

    X, y = prepare_features(df, target_col)
    log.info(f"Feature matrix: {X.shape[1]} features -> {list(X.columns)}")

    seed = config["reproducibility"]["random_seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["ml"]["test_size"], random_state=seed)
    log.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    models, missing_models = build_models(config)
    if missing_models:
        log.warning(f"Models not installed, skipped: {missing_models}")

    results_df, fitted = evaluate_models(
        models, X_train, X_test, y_train, y_test, config["ml"]["cv_folds"], seed, log)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    best_preds = best_model.predict(X_test)
    log.info(f"Best model: {best_name} (Test R²={results_df.iloc[0]['test_r2']:.4f})")

    # --- figures ---
    fig01_model_comparison(results_df, None, config); log.info("Saved module8_01_model_comparison.png")
    fig02_predicted_vs_actual(y_test, best_preds, best_name, None, config); log.info("Saved module8_02_predicted_vs_actual.png")
    fig03_residual_analysis(y_test, best_preds, best_name, config); log.info("Saved module8_03_residual_analysis.png")

    rf_model = fitted.get("Random Forest")
    fig04_feature_importance(rf_model, X.columns, "Random Forest Feature Importance",
                              "module8_04_rf_feature_importance.png", config)
    log.info("Saved module8_04_rf_feature_importance.png")

    xgb_model = fitted.get("XGBoost")
    fig04_feature_importance(xgb_model, X.columns, "XGBoost Feature Importance",
                              "module8_05_xgb_feature_importance.png", config)
    log.info("Saved module8_05_xgb_feature_importance.png")

    shap_result = compute_shap(fitted, X_train, log)
    fig06_shap_bar(shap_result, config); log.info("Saved module8_06_shap_summary_bar.png")
    fig07_shap_beeswarm(shap_result, config); log.info("Saved module8_07_shap_beeswarm.png")
    fig08_shap_dependence(shap_result, config); log.info("Saved module8_08_shap_dependence.png")

    fig09_learning_curves(fitted, X_train, y_train, config["ml"]["cv_folds"], seed, config)
    log.info("Saved module8_09_learning_curves.png")

    fig10_leaderboard(results_df, config); log.info("Saved module8_10_leaderboard.png")

    # --- save best model ---
    models_dir = get_path("models_saved", config)
    model_path = models_dir / "module8_best_model.joblib"
    joblib.dump(best_model, model_path)
    log.info(f"Saved best model -> {model_path}")

    # --- save results table ---
    save_dataframe(results_df, "module8_model_leaderboard.csv", stage="processed", config=config)

    # --- reports ---
    full_report_path = build_full_report(config, results_df, best_name, missing_models,
                                          shap_result, target_col, X.shape[1])
    log.info(f"Full report saved -> {full_report_path}")

    summary = {
        "best_model": best_name,
        "best_test_r2": round(float(results_df.iloc[0]["test_r2"]), 4),
        "best_cv_r2_mean": round(float(results_df.iloc[0]["cv_r2_mean"]), 4),
        "target_r2": config["ml"]["target_r2"],
        "target_achieved": bool(results_df.iloc[0]["test_r2"] >= config["ml"]["target_r2"]),
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "models_skipped_missing_deps": missing_models,
        "shap_available": shap_result is not None,
        "figures_saved": 10,
    }
    write_module_summary(MODULE_NAME, summary, config)

    log.info("=" * 60)
    log.info("MODULE 8 COMPLETE — 10 figures + 2 reports + 1 saved model")
    log.info("=" * 60)
    print("✓ MODULE 8 COMPLETE")

    return best_model, results_df


if __name__ == "__main__":
    run()
