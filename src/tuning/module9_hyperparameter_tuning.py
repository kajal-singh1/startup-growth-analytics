"""
MODULE 9 — HYPERPARAMETER TUNING
============================================================================
OBJECTIVE
  Improve XGBoost (Module 8's winner, Test R²=0.60) by systematically
  searching its hyperparameter space instead of relying on the hand-picked
  defaults used in Module 8 (n_estimators=300, max_depth=4, lr=0.05). Also
  re-check Random Forest (the close runner-up) to confirm XGBoost is still
  the better choice after both get a fair tuning pass.

WHY THIS MODULE IS NEEDED
  Module 8's XGBoost had a CV R² of 0.530 with a STD of 0.260 — that std is
  large relative to the mean, meaning performance is unstable across folds.
  That instability is itself a symptom of under-regularized hyperparameters
  (e.g. no constraint on tree complexity beyond max_depth, no subsampling).
  Tuning isn't just about chasing a higher R² — it's about finding a
  hyperparameter region that's both ACCURATE and STABLE.

THEORY / MATH INTUITION — three search strategies compared head-to-head
  1. GRID SEARCH: exhaustively evaluates every combination on a predefined
     grid via K-fold CV. Guaranteed to find the grid's best point, but cost
     grows multiplicatively with each added parameter/level (the "curse of
     dimensionality" of search itself).
  2. RANDOM SEARCH: samples a fixed budget of random combinations from
     specified distributions. Bergstra & Bengio (2012) showed this usually
     matches or beats grid search at a fraction of the cost, because not
     every hyperparameter matters equally — random sampling explores the
     IMPORTANT dimensions more densely than a grid does for the same budget.
  3. OPTUNA (Bayesian / TPE): a Tree-structured Parzen Estimator models
     which regions of hyperparameter space have produced good scores so far,
     and proposes the NEXT trial's parameters based on that model — each
     trial is informed by all previous ones, unlike grid/random search where
     trials are independent. Typically converges to a good optimum in far
     fewer trials.

ALGORITHMS USED
  GridSearchCV | RandomizedSearchCV | Optuna (TPE sampler) | K-Fold CV

INPUTS
  data/processed/master_features.csv (same as Module 8)
  models/saved/module8_best_model.joblib (the baseline being improved on)

OUTPUTS
  - models/tuned/module9_xgboost_tuned.joblib
  - models/tuned/module9_random_forest_tuned.joblib
  - 10 figures -> outputs/figures/
  - 2 reports  -> outputs/reports/

CONNECTION TO THE NEXT MODULE
  Module 11 (Explainable AI) will run SHAP on whichever model wins here
  (tuned XGBoost vs tuned Random Forest), not on Module 8's defaults.

Run via: python scripts/run_module9.py
"""

import sys
import time
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
                    save_dataframe, load_dataframe, write_module_summary, get_path)
from ml.module8_machine_learning import prepare_features

from sklearn.model_selection import (train_test_split, KFold, cross_val_score,
                                      GridSearchCV, RandomizedSearchCV, learning_curve)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform

MODULE_NAME = "module9_hyperparameter_tuning"


# --------------------------------------------------------------------------- #
# DATA + BASELINE
# --------------------------------------------------------------------------- #
def load_data_and_split(config):
    df = load_dataframe("master_features.csv", stage="processed", config=config)
    target_col = config["data"]["target_variable"]
    X, y = prepare_features(df, target_col)
    seed = config["reproducibility"]["random_seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["ml"]["test_size"], random_state=seed)
    return X, y, X_train, X_test, y_train, y_test


def evaluate(model, X_train, X_test, y_train, y_test, cv, fit=True):
    if fit:
        model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
    preds = model.predict(X_test)
    return {
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "test_r2": round(float(r2_score(y_test, preds)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "test_mae": round(float(mean_absolute_error(y_test, preds)), 4),
    }, preds


# --------------------------------------------------------------------------- #
# SEARCH STRATEGY 1 — GRID SEARCH
# --------------------------------------------------------------------------- #
def run_grid_search(X_train, y_train, cv, seed, n_jobs, log):
    from xgboost import XGBRegressor
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.7, 1.0],
    }
    n_combos = np.prod([len(v) for v in param_grid.values()])
    log.info(f"Grid Search: {n_combos} combinations x {cv.n_splits} folds = {n_combos * cv.n_splits} fits")

    t0 = time.time()
    search = GridSearchCV(
        XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0),
        param_grid, cv=cv, scoring="r2", n_jobs=n_jobs)
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info(f"Grid Search done in {elapsed:.1f}s. Best CV R²={search.best_score_:.4f}  "
              f"params={search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_, elapsed


# --------------------------------------------------------------------------- #
# SEARCH STRATEGY 2 — RANDOM SEARCH
# --------------------------------------------------------------------------- #
def run_random_search(X_train, y_train, cv, seed, n_jobs, n_iter, log):
    from xgboost import XGBRegressor
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(2, 6),
        "learning_rate": uniform(0.01, 0.25),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 8),
        "reg_alpha": uniform(0, 2.0),
        "reg_lambda": uniform(0.5, 3.0),
    }
    log.info(f"Random Search: {n_iter} sampled combinations x {cv.n_splits} folds")

    t0 = time.time()
    search = RandomizedSearchCV(
        XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0),
        param_dist, n_iter=n_iter, cv=cv, scoring="r2", n_jobs=n_jobs, random_state=seed)
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info(f"Random Search done in {elapsed:.1f}s. Best CV R²={search.best_score_:.4f}  "
              f"params={search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_, elapsed


# --------------------------------------------------------------------------- #
# SEARCH STRATEGY 3 — OPTUNA (TPE / BAYESIAN)
# --------------------------------------------------------------------------- #
def run_optuna_search(X_train, y_train, cv, seed, n_jobs, n_trials, log):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("optuna not installed - skipping Optuna search. Install with: pip install optuna")
        return None, None, None, None, None

    from xgboost import XGBRegressor

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 3.5),
        )
        model = XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0, **params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        return scores.mean()

    t0 = time.time()
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best_model = XGBRegressor(random_state=seed, n_jobs=n_jobs, verbosity=0, **study.best_params)
    log.info(f"Optuna done in {elapsed:.1f}s over {n_trials} trials. Best CV R²={study.best_value:.4f}  "
              f"params={study.best_params}")
    return best_model, study.best_params, study.best_value, elapsed, study


# --------------------------------------------------------------------------- #
# RANDOM FOREST — QUICK TUNING PASS (sanity check: is XGBoost still the winner?)
# --------------------------------------------------------------------------- #
def tune_random_forest(X_train, y_train, cv, seed, n_jobs, n_iter, log):
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 12),
        "min_samples_leaf": randint(1, 8),
        "max_features": uniform(0.4, 0.6),
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
        param_dist, n_iter=n_iter, cv=cv, scoring="r2", n_jobs=n_jobs, random_state=seed)
    search.fit(X_train, y_train)
    log.info(f"Random Forest tuning: Best CV R²={search.best_score_:.4f}  params={search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


# --------------------------------------------------------------------------- #
# FIGURES
# --------------------------------------------------------------------------- #
def fig01_search_comparison(results, config):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results.keys())
    means = [results[n]["cv_r2_mean"] for n in names]
    stds = [results[n]["cv_r2_std"] for n in names]
    colors = ["#94a3b8", "#2563eb", "#16a34a", "#dc2626"][:len(names)]
    ax.bar(names, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("Cross-validated R²"); ax.set_title("XGBoost — Baseline vs Tuning Strategies (CV R²)")
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "module9_01_search_comparison.png", config)


def fig02_optuna_history(study, config):
    if study is None:
        _placeholder("module9_02_optuna_history.png",
                     "Optuna not available — see report for install steps.", config)
        return
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(values) + 1), values, "o", alpha=0.4, color="#94a3b8", label="Trial CV R²")
    ax.plot(range(1, len(values) + 1), best_so_far, color="#dc2626", lw=2, label="Best so far")
    ax.set_xlabel("Trial number"); ax.set_ylabel("CV R²")
    ax.set_title("Optuna Optimization History"); ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, "module9_02_optuna_history.png", config)


def fig03_optuna_param_importance(study, config):
    if study is None:
        _placeholder("module9_03_optuna_param_importance.png",
                     "Optuna not available — see report for install steps.", config)
        return
    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
    except Exception as e:
        _placeholder("module9_03_optuna_param_importance.png", f"Could not compute param importance: {e}", config)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(importance.keys())[::-1]
    vals = list(importance.values())[::-1]
    ax.barh(names, vals, color="#7c3aed")
    ax.set_title("Optuna — Hyperparameter Importance"); ax.set_xlabel("Relative importance")
    save_figure(fig, "module9_03_optuna_param_importance.png", config)


def fig04_best_params_table(grid_params, random_params, optuna_params, config):
    all_keys = sorted(set(grid_params) | set(random_params) | set(optuna_params or {}))
    rows = []
    for k in all_keys:
        rows.append([k, _fmt(grid_params.get(k)), _fmt(random_params.get(k)), _fmt((optuna_params or {}).get(k))])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Hyperparameter", "Grid Search", "Random Search", "Optuna"],
                    loc="center", cellLoc="center", colWidths=[0.32, 0.22, 0.22, 0.22])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for j in range(4):
        tbl[0, j].set_facecolor("#1f2937"); tbl[0, j].set_text_props(color="white", weight="bold")
    ax.set_title("Best Hyperparameters by Search Method", pad=16)
    save_figure(fig, "module9_04_best_hyperparameters_table.png", config)


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def fig05_predicted_vs_actual(y_test, preds, model_name, config):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, preds, alpha=0.7, color="#16a34a", edgecolor="white")
    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual growth rate (%)"); ax.set_ylabel("Predicted growth rate (%)")
    ax.set_title(f"Predicted vs Actual — {model_name} (Tuned)")
    ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, "module9_05_predicted_vs_actual_tuned.png", config)


def fig06_residuals_before_after(y_test, preds_before, preds_after, config):
    res_before = y_test.values - preds_before
    res_after = y_test.values - preds_after
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].hist(res_before, bins=12, color="#94a3b8", edgecolor="white")
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_title(f"Before Tuning (std={res_before.std():.2f})"); axes[0].set_xlabel("Residual")
    axes[1].hist(res_after, bins=12, color="#16a34a", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title(f"After Tuning (std={res_after.std():.2f})"); axes[1].set_xlabel("Residual")
    fig.suptitle("Residual Spread — Before vs After Tuning", y=1.02)
    save_figure(fig, "module9_06_residuals_before_after.png", config)


def fig07_rf_tuning_comparison(rf_baseline_res, rf_tuned_res, config):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Baseline RF", "Tuned RF"]
    means = [rf_baseline_res["cv_r2_mean"], rf_tuned_res["cv_r2_mean"]]
    stds = [rf_baseline_res["cv_r2_std"], rf_tuned_res["cv_r2_std"]]
    ax.bar(labels, means, yerr=stds, capsize=5, color=["#94a3b8", "#2563eb"])
    ax.set_ylabel("Cross-validated R²"); ax.set_title("Random Forest — Before vs After Tuning")
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "module9_07_rf_tuning_comparison.png", config)


def fig08_final_leaderboard(leaderboard_df, config):
    fig, ax = plt.subplots(figsize=(10, 1 + 0.55 * len(leaderboard_df)))
    ax.axis("off")
    tbl = ax.table(cellText=leaderboard_df.values, colLabels=leaderboard_df.columns,
                    loc="center", cellLoc="center", colWidths=[0.32, 0.17, 0.14, 0.13, 0.14, 0.14])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for j in range(len(leaderboard_df.columns)):
        tbl[0, j].set_facecolor("#1f2937"); tbl[0, j].set_text_props(color="white", weight="bold")
        tbl[1, j].set_facecolor("#dcfce7")
    ax.set_title("Final Leaderboard — All Models, Baseline + Tuned", pad=16)
    save_figure(fig, "module9_08_final_leaderboard.png", config)


def fig09_learning_curve_before_after(baseline_model, tuned_model, X_train, y_train, cv, seed, config):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, label, color in [(baseline_model, "Baseline XGBoost", "#94a3b8"),
                                  (tuned_model, "Tuned XGBoost", "#dc2626")]:
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, scoring="r2",
            train_sizes=np.linspace(0.3, 1.0, 5), random_state=seed)
        ax.plot(sizes, val_scores.mean(axis=1), marker="o", color=color, label=f"{label} (val)")
        ax.plot(sizes, train_scores.mean(axis=1), marker="o", ls="--", color=color, alpha=0.5,
                label=f"{label} (train)")
    ax.set_xlabel("Training set size"); ax.set_ylabel("R²")
    ax.set_title("Learning Curves — Before vs After Tuning")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, "module9_09_learning_curve_before_after.png", config)


def fig10_overfit_gap(results, config):
    """Train-CV gap as a proxy for overfitting: a smaller gap after tuning
    means the model generalizes more consistently, not just scores higher."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    gaps = [results[n].get("train_cv_gap", np.nan) for n in names]
    colors = ["#94a3b8", "#2563eb", "#16a34a", "#dc2626"][:len(names)]
    ax.bar(names, gaps, color=colors)
    ax.set_ylabel("Train R² − CV R² (lower = less overfitting)")
    ax.set_title("Overfitting Gap — Baseline vs Tuning Strategies")
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "module9_10_overfitting_gap.png", config)


def _placeholder(filename, message, config):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=11, wrap=True)
    ax.axis("off")
    save_figure(fig, filename, config)


# --------------------------------------------------------------------------- #
# REPORT
# --------------------------------------------------------------------------- #
def build_full_report(config, results, best_strategy, best_params, leaderboard_df,
                       rf_baseline_res, rf_tuned_res, rf_params, elapsed_times, optuna_available):
    lines = []
    lines.append("=" * 72)
    lines.append("MODULE 9 — HYPERPARAMETER TUNING — FULL REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    lines.append("\nOBJECTIVE")
    lines.append("-" * 72)
    lines.append("Improve Module 8's XGBoost winner via Grid Search, Random Search, and Optuna,")
    lines.append("and re-check Random Forest to confirm XGBoost remains the better choice.")

    lines.append("\nALGORITHMS USED")
    lines.append("-" * 72)
    lines.append("GridSearchCV | RandomizedSearchCV | Optuna (TPE sampler) | K-Fold CV")
    if not optuna_available:
        lines.append("NOTE: optuna was not installed in this run - install with: pip install optuna")

    lines.append("\nRESULTS — XGBOOST: BASELINE vs EACH TUNING STRATEGY")
    lines.append("-" * 72)
    for name, r in results.items():
        t = f"  ({elapsed_times.get(name, 0):.1f}s)" if name in elapsed_times else ""
        lines.append(f"  {name:15s} CV R²={r['cv_r2_mean']:+.4f}±{r['cv_r2_std']:.4f}   "
                      f"Test R²={r['test_r2']:+.4f}   RMSE={r['test_rmse']:.3f}{t}")

    lines.append(f"\nBest strategy: {best_strategy}")
    lines.append(f"Best hyperparameters found:")
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")

    improvement = results[best_strategy]["test_r2"] - results["Baseline"]["test_r2"]
    lines.append(f"\nTest R² change vs Module 8 baseline: {improvement:+.4f}")

    lines.append("\nRANDOM FOREST — QUICK TUNING SANITY CHECK")
    lines.append("-" * 72)
    lines.append(f"  Baseline RF   CV R²={rf_baseline_res['cv_r2_mean']:+.4f}±{rf_baseline_res['cv_r2_std']:.4f}  "
                  f"Test R²={rf_baseline_res['test_r2']:+.4f}")
    lines.append(f"  Tuned RF      CV R²={rf_tuned_res['cv_r2_mean']:+.4f}±{rf_tuned_res['cv_r2_std']:.4f}  "
                  f"Test R²={rf_tuned_res['test_r2']:+.4f}")
    lines.append(f"  Tuned RF params: {rf_params}")

    overall_winner = leaderboard_df.iloc[0]["model"]
    lines.append(f"\nOVERALL WINNER ACROSS ALL MODELS: {overall_winner}")
    lines.append("\nFINAL LEADERBOARD (all models, baseline + tuned)")
    lines.append("-" * 72)
    lines.append(leaderboard_df.to_string(index=False))

    lines.append("\nLIMITATIONS")
    lines.append("-" * 72)
    lines.append("- Grid Search's grid and Random/Optuna's budgets were kept modest to finish in")
    lines.append("  a reasonable time on a laptop; a larger search budget could find a better point.")
    lines.append("- All three strategies were tuned against the SAME CV folds as the baseline, so")
    lines.append("  improvements reflect genuine generalization gains, not different evaluation luck -")
    lines.append("  but the absolute numbers still inherit Module 8's small-sample-size caveats.")
    lines.append("- Optuna's hyperparameter-importance plot reflects sensitivity WITHIN the ranges")
    lines.append("  searched here, not a universal ranking of XGBoost hyperparameters.")

    lines.append("\nPRACTICAL INTERPRETATION")
    lines.append("-" * 72)
    lines.append("Compare the CV STD (not just the mean) before and after tuning: a real win from")
    lines.append("tuning often shows up as a smaller std (more reliable performance across folds)")
    lines.append("even when the mean R² only moves a little.")

    lines.append("\nBEST PRACTICES APPLIED")
    lines.append("-" * 72)
    lines.append("- All three methods tuned against IDENTICAL CV folds for a fair comparison.")
    lines.append("- Final model selected by CV performance, not by peeking at the test set repeatedly.")
    lines.append("- Random Forest re-tuned too, so the 'XGBoost wins' conclusion isn't assumed -")
    lines.append("  it's re-verified after giving every model a fair shot.")

    lines.append("\nCONNECTION TO THE NEXT MODULE")
    lines.append("-" * 72)
    lines.append(f"Module 11 (Explainable AI) will run SHAP on {overall_winner}, the tuned winner")
    lines.append("from this module, rather than on Module 8's untuned defaults.")

    lines.append("\n" + "=" * 72)
    lines.append("MODULE 9 COMPLETE")
    lines.append("=" * 72)

    reports_dir = get_path("reports", config)
    out_path = reports_dir / "module9_hyperparameter_tuning_report.txt"
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
    log.info("MODULE 9 — HYPERPARAMETER TUNING")
    log.info("=" * 60)

    X, y, X_train, X_test, y_train, y_test = load_data_and_split(config)
    log.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X.shape[1]}")

    seed = config["reproducibility"]["random_seed"]
    n_jobs = config["ml"]["n_jobs"]
    cv = KFold(n_splits=config["ml"]["cv_folds"], shuffle=True, random_state=seed)
    elapsed_times = {}

    # --- baseline (Module 8's untuned XGBoost, rebuilt fresh for a clean comparison) ---
    from xgboost import XGBRegressor
    baseline_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   random_state=seed, n_jobs=n_jobs, verbosity=0)
    baseline_res, baseline_preds = evaluate(baseline_model, X_train, X_test, y_train, y_test, cv)
    baseline_res["train_cv_gap"] = round(
        r2_score(y_train, baseline_model.predict(X_train)) - baseline_res["cv_r2_mean"], 4)
    log.info(f"Baseline XGBoost: CV R²={baseline_res['cv_r2_mean']:.4f}±{baseline_res['cv_r2_std']:.4f}  "
              f"Test R²={baseline_res['test_r2']:.4f}")

    # --- grid search ---
    grid_model, grid_params, grid_cv_r2, grid_time = run_grid_search(X_train, y_train, cv, seed, n_jobs, log)
    elapsed_times["Grid Search"] = grid_time
    grid_res, grid_preds = evaluate(grid_model, X_train, X_test, y_train, y_test, cv, fit=False)
    grid_res["train_cv_gap"] = round(r2_score(y_train, grid_model.predict(X_train)) - grid_res["cv_r2_mean"], 4)

    # --- random search ---
    random_model, random_params, random_cv_r2, random_time = run_random_search(
        X_train, y_train, cv, seed, n_jobs, n_iter=40, log=log)
    elapsed_times["Random Search"] = random_time
    random_res, random_preds = evaluate(random_model, X_train, X_test, y_train, y_test, cv, fit=False)
    random_res["train_cv_gap"] = round(r2_score(y_train, random_model.predict(X_train)) - random_res["cv_r2_mean"], 4)

    # --- optuna ---
    optuna_model, optuna_params, optuna_cv_r2, optuna_time, study = run_optuna_search(
        X_train, y_train, cv, seed, n_jobs, n_trials=40, log=log)
    optuna_available = optuna_model is not None
    if optuna_available:
        elapsed_times["Optuna"] = optuna_time
        optuna_model.fit(X_train, y_train)
        optuna_res, optuna_preds = evaluate(optuna_model, X_train, X_test, y_train, y_test, cv, fit=False)
        optuna_res["train_cv_gap"] = round(
            r2_score(y_train, optuna_model.predict(X_train)) - optuna_res["cv_r2_mean"], 4)
    else:
        optuna_res, optuna_preds = None, None

    results = {"Baseline": baseline_res, "Grid Search": grid_res, "Random Search": random_res}
    models_by_strategy = {"Baseline": baseline_model, "Grid Search": grid_model, "Random Search": random_model}
    if optuna_available:
        results["Optuna"] = optuna_res
        models_by_strategy["Optuna"] = optuna_model

    best_strategy = max(results, key=lambda k: results[k]["cv_r2_mean"])
    best_strategy_for_params = {"Grid Search": grid_params, "Random Search": random_params,
                                 "Optuna": optuna_params, "Baseline": {}}
    best_tuned_model = models_by_strategy[best_strategy]
    best_tuned_preds = best_tuned_model.predict(X_test)
    log.info(f"Best XGBoost tuning strategy: {best_strategy} (CV R²={results[best_strategy]['cv_r2_mean']:.4f})")

    # --- figures 1-6: XGBoost tuning ---
    fig01_search_comparison(results, config); log.info("Saved module9_01_search_comparison.png")
    fig02_optuna_history(study, config); log.info("Saved module9_02_optuna_history.png")
    fig03_optuna_param_importance(study, config); log.info("Saved module9_03_optuna_param_importance.png")
    fig04_best_params_table(grid_params, random_params, optuna_params, config)
    log.info("Saved module9_04_best_hyperparameters_table.png")
    fig05_predicted_vs_actual(y_test, best_tuned_preds, "XGBoost", config)
    log.info("Saved module9_05_predicted_vs_actual_tuned.png")
    fig06_residuals_before_after(y_test, baseline_preds, best_tuned_preds, config)
    log.info("Saved module9_06_residuals_before_after.png")

    # --- random forest re-tune (sanity check) ---
    rf_baseline = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=seed, n_jobs=n_jobs)
    rf_baseline_res, _ = evaluate(rf_baseline, X_train, X_test, y_train, y_test, cv)
    rf_tuned_model, rf_params, rf_cv_r2 = tune_random_forest(X_train, y_train, cv, seed, n_jobs, 40, log)
    rf_tuned_res, _ = evaluate(rf_tuned_model, X_train, X_test, y_train, y_test, cv, fit=False)
    fig07_rf_tuning_comparison(rf_baseline_res, rf_tuned_res, config)
    log.info("Saved module9_07_rf_tuning_comparison.png")

    # --- final leaderboard across everything ---
    leaderboard_rows = [
        {"model": "XGBoost (baseline)", **{k: v for k, v in baseline_res.items() if k != "train_cv_gap"}},
        {"model": f"XGBoost ({best_strategy})", **{k: v for k, v in results[best_strategy].items() if k != "train_cv_gap"}},
        {"model": "Random Forest (baseline)", **rf_baseline_res},
        {"model": "Random Forest (tuned)", **rf_tuned_res},
    ]
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    fig08_final_leaderboard(leaderboard_df, config); log.info("Saved module9_08_final_leaderboard.png")

    fig09_learning_curve_before_after(baseline_model, best_tuned_model, X_train, y_train, cv, seed, config)
    log.info("Saved module9_09_learning_curve_before_after.png")
    fig10_overfit_gap(results, config); log.info("Saved module9_10_overfitting_gap.png")

    # --- save tuned models ---
    tuned_dir = get_path("models_tuned", config)
    joblib.dump(best_tuned_model, tuned_dir / "module9_xgboost_tuned.joblib")
    joblib.dump(rf_tuned_model, tuned_dir / "module9_random_forest_tuned.joblib")
    log.info(f"Saved tuned models -> {tuned_dir}")

    save_dataframe(leaderboard_df, "module9_final_leaderboard.csv", stage="processed", config=config)

    overall_winner = leaderboard_df.iloc[0]["model"]
    full_report_path = build_full_report(
        config, results, best_strategy, best_strategy_for_params[best_strategy], leaderboard_df,
        rf_baseline_res, rf_tuned_res, rf_params, elapsed_times, optuna_available)
    log.info(f"Full report saved -> {full_report_path}")

    summary = {
        "xgboost_baseline_test_r2": baseline_res["test_r2"],
        "xgboost_best_strategy": best_strategy,
        "xgboost_tuned_test_r2": results[best_strategy]["test_r2"],
        "improvement": round(results[best_strategy]["test_r2"] - baseline_res["test_r2"], 4),
        "rf_baseline_test_r2": rf_baseline_res["test_r2"],
        "rf_tuned_test_r2": rf_tuned_res["test_r2"],
        "overall_winner": overall_winner,
        "optuna_available": optuna_available,
        "figures_saved": 10,
    }
    write_module_summary(MODULE_NAME, summary, config)

    log.info("=" * 60)
    log.info(f"MODULE 9 COMPLETE — overall winner: {overall_winner}")
    log.info("=" * 60)
    print("✓ MODULE 9 COMPLETE")

    return leaderboard_df, best_tuned_model, rf_tuned_model


if __name__ == "__main__":
    run()
