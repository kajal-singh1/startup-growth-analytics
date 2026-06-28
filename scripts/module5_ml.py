"""
Module 5: Machine Learning Pipeline
────────────────────────────────────
Models:
  M1  – Baseline: Linear Regression (benchmark)
  M2  – Ridge / Lasso Regression
  M3  – Random Forest Regressor
  M4  – XGBoost Regressor
  M5  – LightGBM Regressor
  M6  – Stacking Ensemble (RF + XGB + LGB → Ridge meta-learner)

Evaluation:
  • 5-Fold Cross-Validation (R², RMSE, MAE)
  • Train/Test split (80/20, time-aware: test = 2023 data)
  • Learning curves
  • SHAP explainability (global + local)

Figures (10):
  01  Model comparison bar chart (CV R²)
  02  Predicted vs Actual (best model)
  03  Residual plot (best model)
  04  Feature importance — Random Forest
  05  Feature importance — XGBoost
  06  SHAP summary plot
  07  SHAP beeswarm
  08  SHAP dependence: internet_penetration
  09  Learning curves (RF & XGB)
  10  Model leaderboard summary table

Report: data/outputs/reports/module5_ml_report.txt
"""

import os, sqlite3, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import (KFold, cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("module5_ml")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
DB   = os.path.join(BASE, "db", "startup_analytics.db")
FIG  = os.path.join(BASE, "data", "outputs", "figures", "module5")
RPT  = os.path.join(BASE, "data", "outputs", "reports", "module5_ml_report.txt")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.dirname(RPT), exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C = {"navy":"1E3A5F", "teal":"0D9488", "red":"C73E1D",
     "gold":"F18F01", "purple":"6D28D9", "green":"1A7A4A"}
CLIST = ["#"+v for v in C.values()]
sns.set_theme(style="whitegrid", font_scale=1.05)

def savefig(name):
    p = os.path.join(FIG, name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {name}")

# ── load & prepare data ───────────────────────────────────────────────────────
conn = sqlite3.connect(DB)
df   = pd.read_sql("SELECT * FROM master_dataset", conn)
conn.close()

# Use confirmed data only
df = df[df["is_partial"] == 0].copy()

FEATURES = [
    "gdp_growth_rate", "gdp_per_capita", "internet_penetration",
    "rd_expenditure_pct", "population_millions", "total_funding_usd_bn",
    "num_deals", "num_unicorns", "funding_per_startup_mn",
]
TARGET = "startup_count"

# Encode period
df["period_enc"] = df["period"].map({"pre": 0, "during": 1, "post": 2})
FEATURES.append("period_enc")

df_ml = df[FEATURES + [TARGET]].dropna()
X = df_ml[FEATURES]
y = df_ml[TARGET]

# Time-aware split: test = year 2023, train = everything before
df_ml2 = df[FEATURES + [TARGET, "year"]].dropna()
X_train = df_ml2[df_ml2["year"] < 2023][FEATURES]
y_train = df_ml2[df_ml2["year"] < 2023][TARGET]
X_test  = df_ml2[df_ml2["year"] == 2023][FEATURES]
y_test  = df_ml2[df_ml2["year"] == 2023][TARGET]

log.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
log.info(f"Features: {FEATURES}")

report_lines = []
def h(t): report_lines.extend(["\n"+"═"*68, f"  {t}", "═"*68])
def r(l=""): report_lines.append(l)

h("MODULE 5 — MACHINE LEARNING PIPELINE")
r(f"Dataset   : {len(df_ml)} rows × {len(FEATURES)} features")
r(f"Target    : {TARGET}")
r(f"Train set : {len(X_train)} rows (2015-2022)")
r(f"Test set  : {len(X_test)} rows (2023)")
r(f"Features  : {', '.join(FEATURES)}")

# ── define models ─────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Lasso Regression":  Lasso(alpha=10.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8,
                                               min_samples_leaf=2, random_state=42),
    "XGBoost":           XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=42, verbosity=0),
    "LightGBM":          LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, verbose=-1),
}

# Stacking ensemble
estimators_stack = [
    ("rf",  RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
    ("xgb", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         random_state=42, verbosity=0)),
    ("lgb", LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                          random_state=42, verbose=-1)),
]
models["Stacking Ensemble"] = StackingRegressor(
    estimators=estimators_stack,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

# ── cross-validation ──────────────────────────────────────────────────────────
h("5-FOLD CROSS-VALIDATION RESULTS")
r(f"{'Model':<22} {'CV R² (mean)':>12} {'CV R² (std)':>12} {'CV RMSE':>12}")
r("-"*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    r2_scores  = cross_val_score(model, X, y, cv=kf, scoring="r2")
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                           scoring="neg_mean_squared_error"))
    cv_results[name] = {
        "r2_mean": r2_scores.mean(), "r2_std": r2_scores.std(),
        "rmse_mean": rmse_scores.mean()
    }
    r(f"{name:<22} {r2_scores.mean():>12.4f} {r2_scores.std():>12.4f} {rmse_scores.mean():>12.1f}")

r()
best_name = max(cv_results, key=lambda k: cv_results[k]["r2_mean"])
r(f"  Best model (CV): {best_name}  R²={cv_results[best_name]['r2_mean']:.4f}")

# ── train/test evaluation ─────────────────────────────────────────────────────
h("TRAIN/TEST EVALUATION (2023 holdout)")
r(f"{'Model':<22} {'Train R²':>10} {'Test R²':>10} {'Test RMSE':>12} {'Test MAE':>10}")
r("-"*66)

test_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test,  y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae  = mean_absolute_error(y_test, y_pred_test)
    test_results[name] = {"train_r2": train_r2, "test_r2": test_r2,
                           "test_rmse": test_rmse, "test_mae": test_mae,
                           "y_pred": y_pred_test}
    r(f"{name:<22} {train_r2:>10.4f} {test_r2:>10.4f} {test_rmse:>12.1f} {test_mae:>10.1f}")

best_test = max(test_results, key=lambda k: test_results[k]["test_r2"])
r(f"\n  Best model (test): {best_test}  Test R²={test_results[best_test]['test_r2']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 01 — Model Comparison (CV R²)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison: Cross-Validation Performance",
             fontsize=14, fontweight="bold", color="#1E3A5F")

names  = list(cv_results.keys())
r2vals = [cv_results[n]["r2_mean"] for n in names]
r2std  = [cv_results[n]["r2_std"]  for n in names]
colors_bar = ["#C73E1D" if n == best_name else "#1E3A5F" for n in names]

bars = axes[0].barh(names, r2vals, xerr=r2std, color=colors_bar, alpha=0.85,
                    error_kw=dict(ecolor="gray", capsize=4))
axes[0].set_xlabel("CV R² Score", fontsize=10)
axes[0].set_title("5-Fold CV R² (mean ± std)", fontsize=11, fontweight="bold")
axes[0].axvline(0.9, color="green", lw=1.2, linestyle="--", alpha=0.6, label="R²=0.90")
axes[0].legend(fontsize=9)
for bar, val in zip(bars, r2vals):
    axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)

# Test RMSE comparison
rmse_vals = [test_results[n]["test_rmse"] for n in names]
colors_rmse = ["#C73E1D" if n == best_test else "#0D9488" for n in names]
bars2 = axes[1].barh(names, rmse_vals, color=colors_rmse, alpha=0.85)
axes[1].set_xlabel("Test RMSE", fontsize=10)
axes[1].set_title("Test RMSE (2023 holdout)", fontsize=11, fontweight="bold")
for bar, val in zip(bars2, rmse_vals):
    axes[1].text(val + 10, bar.get_y() + bar.get_height()/2,
                 f"{val:,.0f}", va="center", fontsize=8)

plt.tight_layout()
savefig("01_model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 02 — Predicted vs Actual (best model)
# ═══════════════════════════════════════════════════════════════════════════════
best_model = models[best_test]
y_pred_best = test_results[best_test]["y_pred"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Predicted vs Actual — {best_test}\n(Test Set: 2023 holdout)",
             fontsize=13, fontweight="bold", color="#1E3A5F")

# scatter
axes[0].scatter(y_test, y_pred_best, color="#1E3A5F", alpha=0.7, s=60, edgecolors="white", lw=0.5)
lim = [min(y_test.min(), y_pred_best.min())*0.95, max(y_test.max(), y_pred_best.max())*1.05]
axes[0].plot(lim, lim, color="#C73E1D", lw=2, linestyle="--", label="Perfect prediction")
axes[0].set_xlabel("Actual Startup Count", fontsize=10)
axes[0].set_ylabel("Predicted Startup Count", fontsize=10)
axes[0].set_title(f"R²={test_results[best_test]['test_r2']:.4f}  RMSE={test_results[best_test]['test_rmse']:,.0f}",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))

# country-level bar
countries_test = df_ml2[df_ml2["year"]==2023]["country"].values if "country" in df_ml2.columns else [f"C{i}" for i in range(len(y_test))]
# Re-fetch with country
df_test_full = df[df["year"]==2023][FEATURES+[TARGET,"country"]].dropna()
y_t2 = df_test_full[TARGET].values
y_p2 = best_model.predict(df_test_full[FEATURES])
ct   = df_test_full["country"].values

x_idx = np.arange(len(ct))
w = 0.35
axes[1].bar(x_idx - w/2, y_t2,  w, label="Actual",    color="#1E3A5F", alpha=0.85)
axes[1].bar(x_idx + w/2, y_p2,  w, label="Predicted", color="#0D9488", alpha=0.85)
axes[1].set_xticks(x_idx)
axes[1].set_xticklabels(ct, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Startup Count", fontsize=10)
axes[1].set_title("Actual vs Predicted by Country (2023)", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:,.0f}"))

plt.tight_layout()
savefig("02_predicted_vs_actual.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 03 — Residuals
# ═══════════════════════════════════════════════════════════════════════════════
residuals = y_test.values - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Residual Analysis — {best_test}", fontsize=13, fontweight="bold", color="#1E3A5F")

axes[0].scatter(y_pred_best, residuals, color="#1E3A5F", alpha=0.7, s=50)
axes[0].axhline(0, color="#C73E1D", lw=1.5, linestyle="--")
axes[0].set_xlabel("Predicted", fontsize=10); axes[0].set_ylabel("Residual", fontsize=10)
axes[0].set_title("Residuals vs Predicted", fontsize=11, fontweight="bold")

axes[1].hist(residuals, bins=12, color="#0D9488", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Residual", fontsize=10); axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].set_title("Residual Distribution", fontsize=11, fontweight="bold")

from scipy import stats as sp_stats
(osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals, dist="norm")
axes[2].scatter(osm, osr, color="#6D28D9", alpha=0.7, s=40)
axes[2].plot(osm, slope*np.array(osm)+intercept, color="#C73E1D", lw=2)
axes[2].set_xlabel("Theoretical Quantiles", fontsize=10)
axes[2].set_ylabel("Sample Quantiles", fontsize=10)
axes[2].set_title("Q-Q Plot of Residuals", fontsize=11, fontweight="bold")

plt.tight_layout()
savefig("03_residual_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 04 — Random Forest Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
rf_model = models["Random Forest"]
rf_model.fit(X_train, y_train)
rf_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors_imp = ["#C73E1D" if v == rf_imp.max() else "#1E3A5F" for v in rf_imp.values]
bars = ax.barh(rf_imp.index, rf_imp.values, color=colors_imp, alpha=0.85)
ax.set_xlabel("Feature Importance (Gini Impurity)", fontsize=10)
ax.set_title("Random Forest — Feature Importance\n(Mean Decrease in Impurity)", 
             fontsize=12, fontweight="bold", color="#1E3A5F")
for bar, val in zip(bars, rf_imp.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
savefig("04_rf_feature_importance.png")

h("RANDOM FOREST FEATURE IMPORTANCE")
for feat, val in rf_imp.sort_values(ascending=False).items():
    r(f"  {feat:<30}  {val:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 05 — XGBoost Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
xgb_model = models["XGBoost"]
xgb_model.fit(X_train, y_train)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors_xgb = ["#F18F01" if v == xgb_imp.max() else "#0D9488" for v in xgb_imp.values]
bars = ax.barh(xgb_imp.index, xgb_imp.values, color=colors_xgb, alpha=0.85)
ax.set_xlabel("Feature Importance (Gain)", fontsize=10)
ax.set_title("XGBoost — Feature Importance (Gain)",
             fontsize=12, fontweight="bold", color="#1E3A5F")
for bar, val in zip(bars, xgb_imp.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
savefig("05_xgb_feature_importance.png")

h("XGBOOST FEATURE IMPORTANCE")
for feat, val in xgb_imp.sort_values(ascending=False).items():
    r(f"  {feat:<30}  {val:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 06 — SHAP Summary Plot
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Computing SHAP values...")
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False,
                  color="#1E3A5F")
ax = plt.gca()
ax.set_title("SHAP Feature Importance — XGBoost\n(Mean |SHAP Value|)",
             fontsize=12, fontweight="bold", color="#1E3A5F")
plt.tight_layout()
savefig("06_shap_summary_bar.png")

h("SHAP MEAN ABSOLUTE VALUES (Global Importance)")
shap_means = np.abs(shap_values).mean(axis=0)
shap_df = pd.Series(shap_means, index=FEATURES).sort_values(ascending=False)
for feat, val in shap_df.items():
    r(f"  {feat:<30}  {val:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 07 — SHAP Beeswarm
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
ax = plt.gca()
ax.set_title("SHAP Beeswarm Plot — XGBoost\n(Impact of Each Feature on Model Output)",
             fontsize=12, fontweight="bold", color="#1E3A5F")
plt.tight_layout()
savefig("07_shap_beeswarm.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 08 — SHAP Dependence: top 2 features
# ═══════════════════════════════════════════════════════════════════════════════
top2 = shap_df.index[:2].tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("SHAP Dependence Plots — Top 2 Features",
             fontsize=13, fontweight="bold", color="#1E3A5F")

for ax, feat in zip(axes, top2):
    feat_idx = FEATURES.index(feat)
    shap.dependence_plot(feat, shap_values, X, ax=ax, show=False,
                         dot_size=30, alpha=0.7)
    ax.set_title(f"SHAP Dependence: {feat}", fontsize=11, fontweight="bold")
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("SHAP Value", fontsize=10)

plt.tight_layout()
savefig("08_shap_dependence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 09 — Learning Curves
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Learning Curves: Random Forest vs XGBoost",
             fontsize=13, fontweight="bold", color="#1E3A5F")

lc_models = [("Random Forest", rf_model, "#1E3A5F"),
             ("XGBoost",       xgb_model, "#0D9488")]

for ax, (name, model, color) in zip(axes, lc_models):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color=color, lw=2, label="Train R²")
    ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color=color)
    ax.plot(train_sizes, val_mean, "s--", color="#C73E1D", lw=2, label="CV R²")
    ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color="#C73E1D")
    ax.set_xlabel("Training Samples", fontsize=10)
    ax.set_ylabel("R² Score", fontsize=10)
    ax.set_title(f"Learning Curve — {name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.4)

plt.tight_layout()
savefig("09_learning_curves.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Model Leaderboard Summary Table
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Module 5 — ML Model Leaderboard", fontsize=14, fontweight="bold", color="#1E3A5F")
ax.axis("off")

leaderboard = []
for name in models.keys():
    cv_r2  = cv_results[name]["r2_mean"]
    cv_std = cv_results[name]["r2_std"]
    t_r2   = test_results[name]["test_r2"]
    t_rmse = test_results[name]["test_rmse"]
    t_mae  = test_results[name]["test_mae"]
    best_flag = "★ BEST" if name == best_test else ""
    leaderboard.append([name, f"{cv_r2:.4f} ± {cv_std:.4f}",
                        f"{t_r2:.4f}", f"{t_rmse:,.0f}", f"{t_mae:,.0f}", best_flag])

leaderboard.sort(key=lambda x: float(x[2]), reverse=True)

tbl = ax.table(
    cellText=leaderboard,
    colLabels=["Model", "CV R² (mean±std)", "Test R²", "Test RMSE", "Test MAE", ""],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for (row_, col_), cell in tbl.get_celld().items():
    if row_ == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")
    elif leaderboard[row_-1][-1] == "★ BEST" if row_ > 0 else False:
        cell.set_facecolor("#FFF3CD")
    elif row_ % 2 == 0:
        cell.set_facecolor("#EAF2FF")
    cell.set_edgecolor("#CCCCCC")

plt.tight_layout()
savefig("10_model_leaderboard.png")

# ── final report summary ──────────────────────────────────────────────────────
h("FINAL SUMMARY")
r(f"  Best model (CV)  : {best_name}  R²={cv_results[best_name]['r2_mean']:.4f}")
r(f"  Best model (test): {best_test}  R²={test_results[best_test]['test_r2']:.4f}  RMSE={test_results[best_test]['test_rmse']:,.0f}")
r(f"\n  Top 3 SHAP features:")
for feat, val in shap_df.head(3).items():
    r(f"    {feat:<30}  mean|SHAP|={val:.4f}")

with open(RPT, "w", encoding="utf-8") as f:
    f.write("MODULE 5 — MACHINE LEARNING REPORT\n")
    f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("\n".join(report_lines))

log.info(f"Report saved → {RPT}")
log.info("="*60)
log.info("MODULE 5 COMPLETE — 10 figures + 1 report")
log.info(f"Location: {FIG}")
for fn in sorted(os.listdir(FIG)):
    log.info(f"  {fn}")
