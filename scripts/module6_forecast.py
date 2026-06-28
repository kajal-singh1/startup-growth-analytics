"""
MODULE 6 — TIME-SERIES FORECASTING
ARIMA, Prophet, and LSTM forecasts of startup growth, with future
projections beyond the observed data range (2015-2024 -> 2025-2029).

Run via: python scripts/run_module6.py
Reads:   ../db/startup_analytics.db   (same db Module 2 created)
Writes:  ../data/outputs/figures/module6/*.png
         ../data/outputs/reports/module6_forecast_report.txt
"""

import os
import sys
import sqlite3
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# PATHS  (mirrors the convention used in module4_stats.py / module5_ml.py)
# --------------------------------------------------------------------------- #
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "..", "db", "startup_analytics.db")
FIG_DIR     = os.path.join(BASE_DIR, "..", "data", "outputs", "figures", "module6")
REPORT_PATH = os.path.join(BASE_DIR, "..", "data", "outputs", "reports", "module6_forecast_report.txt")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# --------------------------------------------------------------------------- #
# CONFIG — change these if auto-detection picks the wrong columns/table
# --------------------------------------------------------------------------- #
FORCE_TABLE  = None   # e.g. "master_dataset" — set if auto-detect guesses wrong
FORCE_YEAR   = None   # e.g. "year"
FORCE_COUNTRY = None  # e.g. "country"
FORCE_TARGET = None   # e.g. "num_startups"
FORECAST_HORIZON_YEARS = 5     # how many years beyond the last observed year to project
TEST_YEARS_HOLDOUT     = 2     # most recent N years held out to score model accuracy

# --------------------------------------------------------------------------- #
# LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | module6_forecast | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("module6_forecast")


# --------------------------------------------------------------------------- #
# 1. LOAD MASTER DATASET — auto-detects table/columns so this script works
#    even if it can't see the exact schema Module 2 created.
# --------------------------------------------------------------------------- #
def _guess_table(conn):
    if FORCE_TABLE:
        return FORCE_TABLE
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    if not tables:
        raise RuntimeError("No tables found in the database.")
    if len(tables) == 1:
        return tables[0]
    # pick the table with the most rows (the joined/master table, not a small lookup table)
    counts = {t: pd.read_sql(f"SELECT COUNT(*) AS n FROM '{t}'", conn)["n"].iloc[0] for t in tables}
    best = max(counts, key=counts.get)
    log.info(f"Multiple tables found {tables} -> auto-selected '{best}' (most rows)")
    return best


def _guess_column(df, forced, candidates_exact, candidates_substring, kind=""):
    if forced:
        return forced
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates_exact:
        if cand in cols_lower:
            return cols_lower[cand]
    for cand in candidates_substring:
        for c_low, c_orig in cols_lower.items():
            if cand in c_low:
                return c_orig
    raise RuntimeError(f"Could not auto-detect the {kind} column among: {list(df.columns)}. "
                        f"Set FORCE_{kind.upper()} at the top of this script.")


def load_master_dataset():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run Module 2 first, or update DB_PATH."
        )
    conn = sqlite3.connect(DB_PATH)
    table = _guess_table(conn)
    df = pd.read_sql(f"SELECT * FROM '{table}'", conn)
    conn.close()

    year_col = _guess_column(df, FORCE_YEAR, ["year", "yr"], ["year"], "year")
    country_col = _guess_column(df, FORCE_COUNTRY, ["country", "nation"], ["country", "nation"], "country")
    target_col = _guess_column(
        df, FORCE_TARGET,
        ["num_startups", "startup_count", "n_startups", "startups", "new_startup_count"],
        ["startup"],
        "target",
    )

    df[year_col] = df[year_col].astype(int)
    df = df.rename(columns={year_col: "_year", country_col: "_country", target_col: "_target"})

    log.info(f"Master dataset: {df.shape[0]} rows x {df.shape[1]} columns (table='{table}')")
    log.info(f"Detected columns -> year='{year_col}'  country='{country_col}'  target='{target_col}'")
    if "period" in [c.lower() for c in df.columns]:
        pcol = [c for c in df.columns if c.lower() == "period"][0]
        log.info(f"period\n{df[pcol].value_counts().to_string()}")

    return df, target_col


# --------------------------------------------------------------------------- #
# 2. BUILD ANNUAL AGGREGATE SERIES (global) + PERIOD LABELS
# --------------------------------------------------------------------------- #
def build_annual_series(df):
    annual = df.groupby("_year").agg(target=("_target", "sum")).reset_index()
    annual = annual.sort_values("_year").reset_index(drop=True)
    annual["ds"] = pd.to_datetime(annual["_year"].astype(str) + "-01-01")
    return annual


def period_label(year):
    if year <= 2019:
        return "pre"
    if year <= 2021:
        return "during"
    return "post"


# --------------------------------------------------------------------------- #
# FIGURE 01 — Historical trend with pandemic period shading
# --------------------------------------------------------------------------- #
def fig01_historical_trend(annual, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(annual["_year"], annual["target"], marker="o", lw=2, color="#2563eb", label="Total startups (observed)")

    pre = annual[annual["_year"] <= 2019]
    during = annual[(annual["_year"] >= 2020) & (annual["_year"] <= 2021)]
    post = annual[annual["_year"] >= 2022]
    if len(during):
        ax.axvspan(during["_year"].min() - 0.5, during["_year"].max() + 0.5, color="grey", alpha=0.15, label="Pandemic period")

    ax.set_title("Historical Global Startup Counts by Year (2015\u20132024)")
    ax.set_xlabel("Year"); ax.set_ylabel("Total startups")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. ARIMA
# --------------------------------------------------------------------------- #
def run_arima(annual, horizon, test_n):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    series = annual.set_index("_year")["target"].astype(float)
    train, test = series.iloc[:-test_n], series.iloc[-test_n:]

    # --- diagnostics figure (ACF/PACF on differenced series) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    diffed = series.diff().dropna()
    max_lag = max(1, len(diffed) // 2 - 1)
    plot_acf(diffed, ax=axes[0], lags=max_lag)
    plot_pacf(diffed, ax=axes[1], lags=max_lag, method="ywm")
    axes[0].set_title("ACF (1st-differenced series)")
    axes[1].set_title("PACF (1st-differenced series)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_arima_diagnostics.png"), dpi=150)
    plt.close(fig)

    # --- order search by AIC over a small grid (sample size is tiny: keep it simple) ---
    best_aic, best_order, best_fit = np.inf, (1, 1, 0), None
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    m = ARIMA(train, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order, best_fit = m.aic, (p, d, q), m
                except Exception:
                    continue
    log.info(f"ARIMA best order={best_order}  AIC={best_aic:.2f}")

    # train-only fit -> honest forecast of the held-out test years (for scoring)
    test_pred = pd.Series(dtype=float)
    if test_n and len(train) >= 3:
        train_fit = ARIMA(train, order=best_order).fit()
        test_fc = train_fit.get_forecast(steps=test_n)
        test_pred = pd.Series(test_fc.predicted_mean.values, index=test.index)

    # full-series fit -> forecast for genuinely unseen future years only
    full_fit = ARIMA(series, order=best_order).fit()
    fc = full_fit.get_forecast(steps=horizon)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=0.2)

    future_years = list(range(series.index.max() + 1, series.index.max() + 1 + horizon))
    fc_mean.index = future_years
    fc_ci.index = future_years

    return {
        "order": best_order, "fit": full_fit,
        "forecast_mean": fc_mean, "forecast_ci": fc_ci,
        "test": test, "test_pred": test_pred,
        "residuals": full_fit.resid,
    }


def fig03_arima_forecast(annual, arima_res, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(annual["_year"], annual["target"], marker="o", color="#2563eb", label="Observed")
    fm = arima_res["forecast_mean"]
    ci = arima_res["forecast_ci"]
    ax.plot(fm.index, fm.values, marker="s", color="#dc2626", label="ARIMA forecast")
    ax.fill_between(fm.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#dc2626", alpha=0.15, label="80% CI")
    ax.axvline(annual["_year"].max(), color="grey", ls="--", lw=1)
    ax.set_title(f"ARIMA{arima_res['order']} Forecast \u2014 Total Startups")
    ax.set_xlabel("Year"); ax.set_ylabel("Total startups")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. PROPHET (optional — degrades gracefully if not installed)
# --------------------------------------------------------------------------- #
def run_prophet(annual, horizon, test_n):
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("Prophet not installed - skipping Prophet model. "
                     "Install with: pip install prophet")
        return None

    pdf = annual[["ds", "target"]].rename(columns={"target": "y"})
    train = pdf.iloc[:-test_n] if test_n else pdf

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                interval_width=0.8)
    m.fit(train)

    future = m.make_future_dataframe(periods=test_n + horizon, freq="YS")
    fcst = m.predict(future)
    return {"model": m, "forecast": fcst, "train": train, "full": pdf}


def fig04_prophet_components(prophet_res, out_path):
    if prophet_res is None:
        _placeholder_figure(out_path, "Prophet not installed \u2014 component plot skipped.\n"
                                       "Install with: pip install prophet")
        return
    fig = prophet_res["model"].plot_components(prophet_res["forecast"])
    fig.suptitle("Prophet Trend Decomposition", y=1.02)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig05_prophet_forecast(annual, prophet_res, out_path):
    if prophet_res is None:
        _placeholder_figure(out_path, "Prophet not installed \u2014 forecast plot skipped.\n"
                                       "Install with: pip install prophet")
        return
    fcst = prophet_res["forecast"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(annual["_year"], annual["target"], marker="o", color="#2563eb", label="Observed")
    fcst_years = fcst["ds"].dt.year
    future_mask = fcst_years > annual["_year"].max() - TEST_YEARS_HOLDOUT
    ax.plot(fcst_years[future_mask], fcst["yhat"][future_mask], marker="s", color="#16a34a", label="Prophet forecast")
    ax.fill_between(fcst_years[future_mask], fcst["yhat_lower"][future_mask], fcst["yhat_upper"][future_mask],
                     color="#16a34a", alpha=0.15, label="80% interval")
    ax.axvline(annual["_year"].max(), color="grey", ls="--", lw=1)
    ax.set_title("Prophet Forecast \u2014 Total Startups")
    ax.set_xlabel("Year"); ax.set_ylabel("Total startups")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _placeholder_figure(out_path, message):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#555",
             transform=ax.transAxes, wrap=True)
    ax.axis("off")
    fig.savefig(out_path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. LSTM (optional — degrades gracefully if TensorFlow not installed)
# --------------------------------------------------------------------------- #
def run_lstm(annual, horizon, test_n, window=2, epochs=200):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        log.warning("TensorFlow not installed - skipping LSTM model. "
                     "Install with: pip install tensorflow-cpu")
        return None

    values = annual["target"].astype(float).values
    scale = values.max() if values.max() > 0 else 1.0
    scaled = values / scale

    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    n_test_samples = max(1, test_n)
    X_train, y_train = X[:-n_test_samples], y[:-n_test_samples]
    X_test, y_test = X[-n_test_samples:], y[-n_test_samples:]

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(16, activation="tanh", input_shape=(window, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0,
                         validation_data=(X_test, y_test) if len(X_test) else None)

    # iterative forecast: roll the window forward through test + future horizon
    last_window = list(scaled[-window:])
    preds_scaled = []
    for _ in range(test_n + horizon):
        x_in = np.array(last_window[-window:]).reshape((1, window, 1))
        p = model.predict(x_in, verbose=0)[0, 0]
        preds_scaled.append(p)
        last_window.append(p)

    preds = np.array(preds_scaled) * scale
    future_years = list(range(annual["_year"].max() - test_n + 1, annual["_year"].max() + 1 + horizon))

    return {"history": history, "preds": preds, "future_years": future_years,
            "test_n": test_n, "model": model}


def fig06_lstm_loss(lstm_res, out_path):
    if lstm_res is None:
        _placeholder_figure(out_path, "TensorFlow not installed \u2014 LSTM loss curve skipped.\n"
                                       "Install with: pip install tensorflow-cpu")
        return
    h = lstm_res["history"].history
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h["loss"], label="Train loss", color="#7c3aed")
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="Validation loss", color="#f59e0b")
    ax.set_title("LSTM Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fig07_lstm_forecast(annual, lstm_res, out_path):
    if lstm_res is None:
        _placeholder_figure(out_path, "TensorFlow not installed \u2014 LSTM forecast skipped.\n"
                                       "Install with: pip install tensorflow-cpu")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(annual["_year"], annual["target"], marker="o", color="#2563eb", label="Observed")
    ax.plot(lstm_res["future_years"], lstm_res["preds"], marker="s", color="#ea580c", label="LSTM forecast")
    ax.axvline(annual["_year"].max() - lstm_res["test_n"], color="grey", ls="--", lw=1)
    ax.set_title("LSTM Forecast \u2014 Total Startups")
    ax.set_xlabel("Year"); ax.set_ylabel("Total startups")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. MODEL COMPARISON OVERLAY + ACCURACY METRICS
# --------------------------------------------------------------------------- #
def _metrics(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if len(actual) == 0:
        return dict(rmse=np.nan, mae=np.nan, mape=np.nan)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    mape = float(np.mean(np.abs((actual - pred) / np.where(actual == 0, np.nan, actual))) * 100)
    return dict(rmse=rmse, mae=mae, mape=mape)


def compute_all_metrics(annual, arima_res, prophet_res, lstm_res, test_n):
    test_years = annual["_year"].iloc[-test_n:].values if test_n else np.array([])
    actual = annual["target"].iloc[-test_n:].values if test_n else np.array([])

    out = {}
    if len(arima_res["test_pred"]):
        out["ARIMA"] = _metrics(actual, arima_res["test_pred"].reindex(test_years).values)
    if prophet_res is not None:
        fy = prophet_res["forecast"]["ds"].dt.year
        pmask = fy.isin(test_years)
        ordered = prophet_res["forecast"][pmask].set_index(fy[pmask])["yhat"].reindex(test_years).values
        out["Prophet"] = _metrics(actual, ordered)
    if lstm_res is not None:
        out["LSTM"] = _metrics(actual, lstm_res["preds"][:test_n])
    return out, test_years, actual


def fig08_model_comparison(annual, arima_res, prophet_res, lstm_res, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(annual["_year"], annual["target"], marker="o", lw=2.5, color="#111827", label="Observed", zorder=5)

    fm = arima_res["forecast_mean"]
    ax.plot(fm.index, fm.values, marker="s", color="#dc2626", label="ARIMA")

    if prophet_res is not None:
        fcst = prophet_res["forecast"]
        fy = fcst["ds"].dt.year
        mask = fy > annual["_year"].max() - TEST_YEARS_HOLDOUT
        ax.plot(fy[mask], fcst["yhat"][mask], marker="^", color="#16a34a", label="Prophet")

    if lstm_res is not None:
        ax.plot(lstm_res["future_years"], lstm_res["preds"], marker="D", color="#ea580c", label="LSTM")

    ax.axvline(annual["_year"].max(), color="grey", ls="--", lw=1)
    ax.set_title("Model Comparison \u2014 All Forecasts Overlaid")
    ax.set_xlabel("Year"); ax.set_ylabel("Total startups")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def fig09_metrics_table(metrics, out_path):
    rows = [[name, f"{m['rmse']:.1f}", f"{m['mae']:.1f}",
             f"{m['mape']:.1f}%" if not np.isnan(m["mape"]) else "n/a"]
            for name, m in metrics.items()]
    if not rows:
        rows = [["(no model produced test-period predictions)", "-", "-", "-"]]

    fig, ax = plt.subplots(figsize=(8, 1 + 0.6 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Model", "RMSE", "MAE", "MAPE"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.8)
    for j in range(4):
        tbl[0, j].set_facecolor("#1f2937")
        tbl[0, j].set_text_props(color="white", weight="bold")
    ax.set_title(f"Forecast Accuracy on Held-Out Test Years (last {TEST_YEARS_HOLDOUT} years)", pad=20)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# 7. COUNTRY-LEVEL NEXT-YEAR FORECAST (simple CAGR extrapolation, top 5)
# --------------------------------------------------------------------------- #
def fig10_country_forecast(df, out_path):
    last_year = df["_year"].max()
    totals_by_country = df[df["_year"] == last_year].groupby("_country")["_target"].sum()
    top5 = totals_by_country.sort_values(ascending=False).head(5).index.tolist()

    next_year_vals = {}
    for c in top5:
        series = df[df["_country"] == c].groupby("_year")["_target"].sum().sort_index()
        if len(series) >= 4 and series.iloc[-4] > 0:
            cagr = (series.iloc[-1] / series.iloc[-4]) ** (1 / 3) - 1
        else:
            cagr = 0.0
        next_year_vals[c] = series.iloc[-1] * (1 + cagr)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(top5))
    current_vals = [totals_by_country[c] for c in top5]
    forecast_vals = [next_year_vals[c] for c in top5]
    width = 0.35
    ax.bar(x - width / 2, current_vals, width, label=f"{last_year} (actual)", color="#2563eb")
    ax.bar(x + width / 2, forecast_vals, width, label=f"{last_year + 1} (forecast)", color="#dc2626")
    ax.set_xticks(x); ax.set_xticklabels(top5, rotation=20, ha="right")
    ax.set_ylabel("Total startups")
    ax.set_title(f"Top 5 Countries \u2014 {last_year} vs {last_year+1} Forecast (CAGR extrapolation)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return top5, current_vals, forecast_vals


# --------------------------------------------------------------------------- #
# 8. REPORT
# --------------------------------------------------------------------------- #
def build_report(annual, arima_res, prophet_res, lstm_res, metrics, country_info, target_col):
    last_year = int(annual["_year"].max())
    lines = []
    lines.append("=" * 70)
    lines.append("MODULE 6 — TIME-SERIES FORECASTING REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"\nTarget variable forecast: '{target_col}' (aggregated globally by year)")
    lines.append(f"Historical range: {int(annual['_year'].min())}-{last_year}  ({len(annual)} yearly points)")
    lines.append(f"Held-out test window: last {TEST_YEARS_HOLDOUT} year(s)")
    lines.append(f"Forecast horizon: {last_year+1}-{last_year+FORECAST_HORIZON_YEARS}")

    lines.append("\n" + "-" * 70)
    lines.append("MODEL 1 — ARIMA")
    lines.append("-" * 70)
    lines.append(f"Best order (p,d,q): {arima_res['order']}  |  AIC: {arima_res['fit'].aic:.2f}")
    lines.append("Forecast (mean):")
    for yr, val in arima_res["test_pred"].items():
        lines.append(f"  {yr} (test): {val:,.0f}")
    for yr, val in arima_res["forecast_mean"].items():
        lines.append(f"  {yr} (future): {val:,.0f}")

    lines.append("\n" + "-" * 70)
    lines.append("MODEL 2 — Prophet")
    lines.append("-" * 70)
    if prophet_res is None:
        lines.append("SKIPPED — Prophet not installed. Install with: pip install prophet")
    else:
        fy = prophet_res["forecast"]["ds"].dt.year
        mask = fy > last_year - TEST_YEARS_HOLDOUT
        for yr, val in zip(fy[mask], prophet_res["forecast"]["yhat"][mask]):
            tag = " (test)" if yr <= last_year else " (future)"
            lines.append(f"  {yr}{tag}: {val:,.0f}")

    lines.append("\n" + "-" * 70)
    lines.append("MODEL 3 — LSTM")
    lines.append("-" * 70)
    if lstm_res is None:
        lines.append("SKIPPED — TensorFlow not installed. Install with: pip install tensorflow-cpu")
    else:
        for yr, val in zip(lstm_res["future_years"], lstm_res["preds"]):
            tag = " (test)" if yr <= last_year else " (future)"
            lines.append(f"  {yr}{tag}: {val:,.0f}")

    lines.append("\n" + "-" * 70)
    lines.append("ACCURACY ON HELD-OUT TEST YEARS")
    lines.append("-" * 70)
    if metrics:
        for name, m in metrics.items():
            mape_str = f"{m['mape']:.1f}%" if not np.isnan(m["mape"]) else "n/a"
            lines.append(f"  {name:10s} RMSE={m['rmse']:>10,.1f}   MAE={m['mae']:>10,.1f}   MAPE={mape_str}")
        best_model = min(metrics, key=lambda k: metrics[k]["rmse"])
        lines.append(f"\n  Lowest RMSE on test data: {best_model}")
    else:
        lines.append("  No model had test-period predictions to score.")

    lines.append("\n" + "-" * 70)
    lines.append("COUNTRY-LEVEL NEXT-YEAR FORECAST (CAGR extrapolation, top 5 by volume)")
    lines.append("-" * 70)
    top5, current_vals, forecast_vals = country_info
    for c, cur, fcv in zip(top5, current_vals, forecast_vals):
        pct = (fcv / cur - 1) * 100 if cur else 0
        lines.append(f"  {c:15s} {last_year}: {cur:>10,.0f}   {last_year+1} forecast: {fcv:>10,.0f}  ({pct:+.1f}%)")

    lines.append("\n" + "-" * 70)
    lines.append("CAVEATS")
    lines.append("-" * 70)
    lines.append(f"  - Only {len(annual)} annual observations are available; all forecasts beyond")
    lines.append("    2-3 years carry wide uncertainty regardless of which model is used.")
    lines.append("  - LSTM is trained on a very short series; treat it as illustrative of the")
    lines.append("    methodology rather than a production-grade forecast.")
    lines.append("  - Country-level forecasts use simple CAGR extrapolation, not the full models.")

    lines.append("\n" + "=" * 70)
    lines.append("MODULE 6 COMPLETE")
    lines.append("=" * 70)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Report saved to {REPORT_PATH}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    df, target_col = load_master_dataset()
    annual = build_annual_series(df)

    test_n = min(TEST_YEARS_HOLDOUT, max(1, len(annual) - 4))  # keep at least 4 points to train on
    horizon = FORECAST_HORIZON_YEARS

    fig01_historical_trend(annual, os.path.join(FIG_DIR, "01_historical_trend.png"))

    arima_res = run_arima(annual, horizon, test_n)
    fig03_arima_forecast(annual, arima_res, os.path.join(FIG_DIR, "03_arima_forecast.png"))
    log.info("Saved 01_historical_trend.png")
    log.info("Saved 02_arima_diagnostics.png")
    log.info("Saved 03_arima_forecast.png")

    prophet_res = run_prophet(annual, horizon, test_n)
    fig04_prophet_components(prophet_res, os.path.join(FIG_DIR, "04_prophet_components.png"))
    fig05_prophet_forecast(annual, prophet_res, os.path.join(FIG_DIR, "05_prophet_forecast.png"))
    log.info("Saved 04_prophet_components.png")
    log.info("Saved 05_prophet_forecast.png")

    lstm_res = run_lstm(annual, horizon, test_n)
    fig06_lstm_loss(lstm_res, os.path.join(FIG_DIR, "06_lstm_training_loss.png"))
    fig07_lstm_forecast(annual, lstm_res, os.path.join(FIG_DIR, "07_lstm_forecast.png"))
    log.info("Saved 06_lstm_training_loss.png")
    log.info("Saved 07_lstm_forecast.png")

    fig08_model_comparison(annual, arima_res, prophet_res, lstm_res, os.path.join(FIG_DIR, "08_model_comparison.png"))
    log.info("Saved 08_model_comparison.png")

    metrics, test_years, actual = compute_all_metrics(annual, arima_res, prophet_res, lstm_res, test_n)
    fig09_metrics_table(metrics, os.path.join(FIG_DIR, "09_accuracy_metrics_table.png"))
    log.info("Saved 09_accuracy_metrics_table.png")

    country_info = fig10_country_forecast(df, os.path.join(FIG_DIR, "10_country_level_forecast.png"))
    log.info("Saved 10_country_level_forecast.png")

    build_report(annual, arima_res, prophet_res, lstm_res, metrics, country_info, target_col)

    log.info("=" * 60)
    log.info("MODULE 6 COMPLETE — 10 figures + 1 report")
    log.info(f"Location: {FIG_DIR}")
    for fname in sorted(os.listdir(FIG_DIR)):
        log.info(f"  {fname}")
    print("✓ MODULE 6 COMPLETE")


if __name__ == "__main__":
    main()
