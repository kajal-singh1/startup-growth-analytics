"""
module10_forecasting.py — Time Series Forecasting
===================================================

OBJECTIVE
---------
Forecast startup growth trends to 2027 using two complementary
approaches:
  1. ARIMA   — classical statistical time series model
  2. LSTM    — deep learning sequence model

WHY TWO MODELS
--------------
ARIMA captures linear autocorrelation and trend/seasonality.
LSTM captures non-linear patterns and cross-country signals.
Comparing both gives confidence in the forecast direction.

MATHEMATICAL NOTES
------------------
ARIMA(p, d, q):
  - p = autoregressive order: y_t depends on y_{t-1}...y_{t-p}
  - d = differencing order: removes trend (makes series stationary)
  - q = moving average order: y_t depends on past errors e_{t-1}...
  Stationarity check: ADF test  H0: unit root exists (non-stationary)
  Reject H0 (p < 0.05) → series is stationary → d=0

LSTM (Long Short-Term Memory):
  Forget gate:  f_t = sigma(W_f [h_{t-1}, x_t] + b_f)
  Input gate:   i_t = sigma(W_i [h_{t-1}, x_t] + b_i)
  Cell update:  C_t = f_t * C_{t-1} + i_t * tanh(W_C [h_{t-1}, x_t])
  Output:       h_t = o_t * tanh(C_t)
  Learns long-range dependencies that ARIMA misses.

MAPE (Mean Absolute Percentage Error):
  MAPE = 100/n * sum(|y_i - y_hat_i| / |y_i|)
  Target: MAPE < 15% (per project spec)

FIGURES (10)
------------
 1. Historical trend — all countries startup growth
 2. Stationarity test results (ADF) per country
 3. ARIMA forecast — global aggregate to 2027
 4. ARIMA country-level forecasts — top 6
 5. LSTM training loss curve
 6. LSTM forecast — global aggregate to 2027
 7. ARIMA vs LSTM comparison
 8. Forecast confidence intervals — top 5 countries
 9. Model accuracy table (MAPE, RMSE per country)
10. Forecast heatmap — country × year (2024-2027)

INPUTS
------
- data/master_features.csv  (or startup_ecosystem_raw.csv as fallback)

OUTPUTS
-------
- data/outputs/figures/module10/*.png  (10 figures)
- data/outputs/reports/module10_forecasting_report.txt
- data/forecasts_2027.csv
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
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils import get_logger, get_project_root

logger  = get_logger("module10_forecasting")
ROOT    = get_project_root()

FIG_DIR = ROOT / "data" / "outputs" / "figures" / "module10"
REP_DIR = ROOT / "data" / "outputs" / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
RANDOM_STATE  = 42
FORECAST_YEAR = 2027
COUNTRY_COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12",
    "#9b59b6","#1abc9c","#e67e22","#34495e",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_time_series():
    """
    Load and build a country x year panel with startup_count as the
    primary time series. Falls back through multiple file candidates.
    """
    candidates = [
    ROOT / "data" / "processed" / "master_features.csv",
    ROOT / "data" / "master_features.csv",
    ROOT / "data" / "processed" / "master_clean.csv",
    ROOT / "data" / "interim" / "master_raw.csv",
    ]

    df = None
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded: {path.name}  shape={df.shape}")
            break

    if df is None:
        raise FileNotFoundError("No data file found. Run Module 2 first.")

    # Aggregate to country-year level
    # Detect country column name
    if "country" in df.columns:
        country_col = "country"
    elif "country_name" in df.columns:
        country_col = "country_name"
    elif "country_code" in df.columns:
        country_col = "country_code"
    else:
        raise ValueError(f"No country column found. Columns: {list(df.columns)}")

    # Detect startup count column
    if "startup_count" in df.columns:
        count_col = "startup_count"
    elif "num_startups" in df.columns:
        count_col = "num_startups"
    else:
        raise ValueError(f"No startup count column found. Columns: {list(df.columns)}")

    group_cols = [country_col, "year"]
    agg = df.groupby(group_cols)[count_col].sum().reset_index()
    agg.columns = ["country", "year", "startup_count"]
    agg = agg.sort_values(["country", "year"])

    # Keep countries with full coverage
    year_counts = agg.groupby("country")["year"].count()
    min_years   = year_counts.max() - 1
    valid       = year_counts[year_counts >= min_years].index
    agg         = agg[agg["country"].isin(valid)]

    logger.info(f"Panel: {agg['country'].nunique()} countries, "
                f"years {agg['year'].min()}-{agg['year'].max()}")
    return agg


def build_global_series(agg):
    """Sum across all countries to get a global aggregate series."""
    global_s = agg.groupby("year")["startup_count"].sum()
    logger.info(f"Global series: {len(global_s)} years  "
                f"mean={global_s.mean():.0f}  std={global_s.std():.0f}")
    return global_s


# ─────────────────────────────────────────────────────────────────────────────
# STATIONARITY
# ─────────────────────────────────────────────────────────────────────────────

def adf_test(series, name="series"):
    """Augmented Dickey-Fuller test for stationarity."""
    series = series.dropna()
    # ADF requires variance — skip constant series
    if series.std() == 0 or len(series) < 4:
        logger.warning(f"ADF [{name}]: skipped (constant or too short)")
        return {"stat": 0, "p": 1.0, "stationary": False, "crit": {}}
    result = adfuller(series, autolag="AIC")
    stat, p, _, _, crit, _ = result
    stationary = p < 0.05
    logger.info(f"ADF [{name}]: stat={stat:.4f}  p={p:.4f}  "
                f"{'STATIONARY' if stationary else 'NON-STATIONARY'}")
    return {"stat": stat, "p": p, "stationary": stationary, "crit": crit}


def check_all_countries(agg):
    """Run ADF on every country's startup_count series."""
    results = {}
    for country in agg["country"].unique():
        s = agg[agg["country"] == country].set_index("year")["startup_count"]
        results[country] = adf_test(s, country)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def fit_arima(series, name="series"):
    """
    Auto-select ARIMA order by AIC search over small grid.
    Returns fitted model and best (p,d,q).
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_model = None

    for p, d, q in product(range(3), range(2), range(3)):
        try:
            m = ARIMA(series, order=(p, d, q))
            r = m.fit()
            if r.aic < best_aic:
                best_aic   = r.aic
                best_order = (p, d, q)
                best_model = r
        except Exception:
            pass

    logger.info(f"ARIMA [{name}]: order={best_order}  AIC={best_aic:.2f}")
    return best_model, best_order


def arima_forecast(model, steps, alpha=0.20):
    """Forecast `steps` periods ahead with confidence interval."""
    fc   = model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci   = fc.conf_int(alpha=alpha)   # 80% CI
    # Reset to positional index to avoid date-range errors with integer year index
    mean = pd.Series(mean.values, name="forecast")
    ci   = pd.DataFrame(ci.values, columns=["lower", "upper"])
    return mean, ci


def fit_all_arima(agg, countries):
    """Fit ARIMA for each country and return models + forecasts."""
    last_year     = int(agg["year"].max())
    horizon       = FORECAST_YEAR - last_year
    arima_results = {}

    for country in countries:
        s = agg[agg["country"] == country].set_index("year")["startup_count"]
        if len(s) < 4:
            continue
        try:
            country_last  = int(s.index.max())
            country_horiz = FORECAST_YEAR - country_last
            if country_horiz < 1:
                continue
            model, order = fit_arima(s, country)
            mean, ci     = arima_forecast(model, country_horiz)
            future_years = list(range(country_last + 1, FORECAST_YEAR + 1))
            arima_results[country] = {
                "model":        model,
                "order":        order,
                "history":      s,
                "forecast_mean": pd.Series(mean.values, index=future_years),
                "forecast_ci":   ci,
                "future_years":  future_years,
            }
        except Exception as e:
            import traceback
            logger.warning(f"ARIMA failed for {country}: {e}\n{traceback.format_exc()}")

    logger.info(f"ARIMA fitted for {len(arima_results)} countries")
    return arima_results, horizon


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(series_values, lookback=3):
    """Convert a 1D series into (X, y) sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(series_values)):
        X.append(series_values[i-lookback:i])
        y.append(series_values[i])
    return np.array(X), np.array(y)


def fit_lstm_global(global_series):
    """
    Fit a simple LSTM on the global aggregate series.
    Returns model, history, scaler, and lookback.
    Uses TensorFlow if available, else falls back to ARIMA-style
    linear extrapolation (graceful degradation).
    """
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_STATE)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler

        values = global_series.values.reshape(-1, 1).astype(float)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values).flatten()

        lookback = 3
        X, y     = build_sequences(scaled, lookback)
        X        = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            LSTM(32, input_shape=(lookback, 1), return_sequences=False),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X, y, epochs=80, batch_size=4,
                            verbose=0, validation_split=0.1)

        logger.info(f"LSTM trained: final loss={history.history['loss'][-1]:.6f}")
        return model, history, scaler, lookback, scaled, "lstm"

    except ImportError:
        logger.warning("TensorFlow not available — using linear extrapolation")
        return None, None, None, None, None, "linear"


def lstm_forecast(model, scaler, lookback, scaled, steps, mode="lstm"):
    """
    Roll forward `steps` periods using the LSTM or linear fallback.
    """
    if mode == "linear":
        # Simple linear extrapolation
        x = np.arange(len(scaled) if scaled is not None else 9)
        y_vals = scaled if scaled is not None else np.zeros(9)
        slope, intercept = np.polyfit(x, y_vals, 1)
        return np.array([intercept + slope * (len(x) + i) for i in range(steps)])

    preds = []
    window = list(scaled[-lookback:])
    for _ in range(steps):
        x_in = np.array(window[-lookback:]).reshape(1, lookback, 1)
        p    = model.predict(x_in, verbose=0)[0, 0]
        preds.append(p)
        window.append(p)

    preds_inv = scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()
    return preds_inv


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual, predicted):
    """RMSE, MAE, MAPE on overlapping indices."""
    actual    = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask      = actual != 0
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae  = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}


def backtest_arima(agg, countries, test_years=2):
    """
    Simple backtest: train on all-but-last-N years, predict last N.
    """
    metrics = {}
    for country in countries[:8]:   # limit to 8 for speed
        s = agg[agg["country"] == country].set_index("year")["startup_count"]
        if len(s) < test_years + 3:
            continue
        train = s.iloc[:-test_years]
        test  = s.iloc[-test_years:]
        try:
            model, _ = fit_arima(train, country)
            mean, _  = arima_forecast(model, test_years)
            m = compute_metrics(test.values, mean.values[:test_years])
            metrics[country] = m
            logger.info(f"  Backtest [{country}]: MAPE={m['mape']:.1f}%  "
                        f"RMSE={m['rmse']:.0f}")
        except Exception as e:
            logger.warning(f"Backtest failed [{country}]: {e}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {name}")


def fig1_historical_trend(agg):
    """Historical startup count by country."""
    countries = agg["country"].unique()
    fig, ax   = plt.subplots(figsize=(14, 6))
    for i, country in enumerate(sorted(countries)):
        sub = agg[agg["country"] == country]
        ax.plot(sub["year"], sub["startup_count"],
                marker="o", linewidth=1.8, markersize=4,
                color=COUNTRY_COLORS[i % len(COUNTRY_COLORS)],
                label=country, alpha=0.8)
    ax.axvspan(2019.5, 2021.5, alpha=0.1, color="red", label="COVID-19")
    ax.set_title("Historical Startup Count by Country (2015–2023)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Startup Count")
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    savefig("01_historical_trend.png")


def fig2_adf_results(adf_results):
    """ADF stationarity test results per country."""
    countries = list(adf_results.keys())
    p_vals    = [adf_results[c]["p"] for c in countries]
    colors    = ["#2ecc71" if adf_results[c]["stationary"]
                 else "#e74c3c" for c in countries]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(countries, p_vals, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(0.05, color="black", linestyle="--",
               linewidth=1.5, label="p=0.05 threshold")
    ax.set_title("ADF Stationarity Test — p-value per Country\n"
                 "Green = stationary (p<0.05), Red = non-stationary",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("ADF p-value"); ax.set_xlabel("")
    ax.legend()
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    savefig("02_adf_stationarity.png")


def fig3_arima_global(global_series, arima_results):
    """ARIMA forecast of global aggregate."""
    last_year    = int(global_series.index.max())
    horizon      = int(FORECAST_YEAR - last_year)
    global_model, _ = fit_arima(global_series, "Global")
    mean, ci        = arima_forecast(global_model, horizon)
    future_years    = list(range(last_year + 1, FORECAST_YEAR + 1))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(global_series.index, global_series.values,
            "o-", color="#3498db", linewidth=2.5, label="Historical", markersize=6)
    ax.plot(future_years, mean.values,
            "o--", color="#e74c3c", linewidth=2.5, label="ARIMA Forecast", markersize=6)
    ax.fill_between(future_years, ci["lower"].values, ci["upper"].values,
                    alpha=0.2, color="#e74c3c", label="80% CI")
    ax.axvline(last_year + 0.5, color="gray", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_title(f"ARIMA Forecast — Global Startup Count to {FORECAST_YEAR}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Startup Count")
    ax.legend()
    plt.tight_layout()
    savefig("03_arima_global_forecast.png")
    return mean, ci


def fig4_arima_countries(arima_results):
    """ARIMA country-level forecasts — top 6 by startup count."""
    top6    = sorted(arima_results.keys())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, country in zip(axes.flat, top6):
        r    = arima_results[country]
        hist = r["history"]
        mean = r["forecast_mean"]
        ax.plot(hist.index, hist.values, "o-", color="#3498db",
                linewidth=2, markersize=4, label="Historical")
        ax.plot(r["future_years"], mean.values, "o--", color="#e74c3c",
                linewidth=2, markersize=4, label="Forecast")
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Startups")
        ax.legend(fontsize=7)
    fig.suptitle(f"ARIMA Country Forecasts to {FORECAST_YEAR}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("04_arima_country_forecasts.png")


def fig5_lstm_loss(hist_obj, mode):
    """LSTM training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if mode == "lstm" and hist_obj is not None:
        ax.plot(hist_obj.history["loss"], color="#3498db",
                linewidth=2, label="Train Loss")
        if "val_loss" in hist_obj.history:
            ax.plot(hist_obj.history["val_loss"], color="#e74c3c",
                    linewidth=2, label="Val Loss")
        ax.set_title("LSTM Training Loss Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.legend()
    else:
        ax.text(0.5, 0.5,
                "LSTM not available\nUsing linear extrapolation",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="gray")
        ax.set_title("LSTM Training (Fallback Mode)", fontsize=12)
    plt.tight_layout()
    savefig("05_lstm_loss.png")


def fig6_lstm_global(global_series, lstm_preds, mode):
    """LSTM forecast of global aggregate."""
    last_year    = int(global_series.index.max())
    future_years = list(range(last_year + 1, FORECAST_YEAR + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(global_series.index, global_series.values,
            "o-", color="#3498db", linewidth=2.5,
            label="Historical", markersize=6)
    label = "LSTM Forecast" if mode == "lstm" else "Linear Extrapolation"
    ax.plot(future_years, lstm_preds, "o--", color="#2ecc71",
            linewidth=2.5, label=label, markersize=6)
    ax.axvline(last_year + 0.5, color="gray", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_title(f"{label} — Global Startup Count to {FORECAST_YEAR}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Startup Count")
    ax.legend()
    plt.tight_layout()
    savefig("06_lstm_global_forecast.png")


def fig7_arima_vs_lstm(global_series, arima_mean, lstm_preds):
    """Overlay ARIMA and LSTM forecasts."""
    last_year    = int(global_series.index.max())
    future_years = list(range(last_year + 1, FORECAST_YEAR + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(global_series.index, global_series.values,
            "o-", color="#3498db", linewidth=2.5,
            label="Historical", markersize=6)
    ax.plot(future_years, arima_mean.values,
            "o--", color="#e74c3c", linewidth=2,
            label="ARIMA", markersize=5)
    ax.plot(future_years, lstm_preds,
            "s--", color="#2ecc71", linewidth=2,
            label="LSTM / Linear", markersize=5)
    ax.axvline(last_year + 0.5, color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label="Forecast start")
    ax.set_title(f"ARIMA vs LSTM — Global Startup Count Forecast to {FORECAST_YEAR}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Startup Count")
    ax.legend()
    plt.tight_layout()
    savefig("07_arima_vs_lstm.png")


def fig8_confidence_intervals(arima_results):
    """Confidence intervals for top 5 countries."""
    top5  = sorted(arima_results.keys())[:5]
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for ax, country in zip(axes, top5):
        r    = arima_results[country]
        hist = r["history"]
        mean = r["forecast_mean"]
        ci   = r["forecast_ci"]
        ax.plot(hist.index, hist.values, "o-", color="#3498db",
                linewidth=1.8, markersize=4)
        ax.plot(r["future_years"], mean.values, "o--",
                color="#e74c3c", linewidth=1.8, markersize=4)
        ax.fill_between(r["future_years"],
                        ci["lower"].values, ci["upper"].values,
                        alpha=0.25, color="#e74c3c")
        ax.set_title(country, fontsize=9, fontweight="bold")
        ax.set_xlabel("Year"); ax.tick_params(labelsize=7)
    fig.suptitle(f"Forecast with 80% Confidence Intervals — Top 5 Countries",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("08_confidence_intervals.png")


def fig9_accuracy_table(backtest_metrics):
    """Model accuracy table as a figure."""
    if not backtest_metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Backtest metrics not available",
                ha="center", va="center", transform=ax.transAxes)
        savefig("09_accuracy_table.png")
        return

    rows = [[c,
             f"{backtest_metrics[c]['mape']:.1f}%",
             f"{backtest_metrics[c]['rmse']:.0f}",
             f"{backtest_metrics[c]['mae']:.0f}",
             "PASS" if backtest_metrics[c]['mape'] < 15 else "FAIL"]
            for c in backtest_metrics]

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Country", "MAPE", "RMSE", "MAE", "Spec (MAPE<15%)"],
        cellLoc="center", loc="center",
        colColours=["#2c3e50"] * 5,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        elif rows[r-1][-1] == "PASS":
            cell.set_facecolor("#d5f5e3")
        else:
            cell.set_facecolor("#fadbd8")
    ax.set_title("ARIMA Backtest Accuracy — 2-Year Holdout",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    savefig("09_accuracy_table.png")


def fig10_forecast_heatmap(arima_results):
    """Heatmap of forecasted startup counts — country × year."""
    rows = {}
    for country, r in arima_results.items():
        rows[country] = {yr: val for yr, val in
                         zip(r["future_years"], r["forecast_mean"].values)}
    fc_df = pd.DataFrame(rows).T.sort_index()

    if fc_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No forecast data", ha="center", va="center")
        savefig("10_forecast_heatmap.png")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(fc_df.columns) * 1.5),
                                    max(5, len(fc_df) * 0.5 + 1)))
    sns.heatmap(fc_df.round(0).astype(int), annot=True, fmt="d",
                cmap="YlOrRd", linewidths=0.4, ax=ax,
                cbar_kws={"label": "Forecasted Startup Count"})
    ax.set_title(f"Forecasted Startup Count — Country x Year ({fc_df.columns.min()}–{FORECAST_YEAR})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Country")
    plt.tight_layout()
    savefig("10_forecast_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FORECASTS
# ─────────────────────────────────────────────────────────────────────────────

def save_forecasts(arima_results, global_series, arima_global_mean, lstm_preds):
    rows = []
    last_year    = int(global_series.index.max())
    future_years = list(range(last_year + 1, FORECAST_YEAR + 1))

    # Country-level ARIMA
    for country, r in arima_results.items():
        for yr, val in zip(r["future_years"], r["forecast_mean"].values):
            rows.append({
                "country": country,
                "year":    yr,
                "model":   "ARIMA",
                "forecast_startup_count": round(float(val), 1),
            })

    # Global LSTM
    for yr, val in zip(future_years, lstm_preds):
        rows.append({
            "country": "GLOBAL",
            "year":    yr,
            "model":   "LSTM",
            "forecast_startup_count": round(float(val), 1),
        })

    fc_df = pd.DataFrame(rows)
    path  = ROOT / "data" / "forecasts_2027.csv"
    fc_df.to_csv(path, index=False)
    logger.info(f"Forecasts saved: {path}  ({len(fc_df)} rows)")
    return fc_df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(adf_results, backtest_metrics, arima_results,
                 horizon, lstm_mode):
    n_stationary = sum(1 for r in adf_results.values() if r["stationary"])
    avg_mape = (np.mean([m["mape"] for m in backtest_metrics.values()])
                if backtest_metrics else float("nan"))

    lines = [
        "=" * 60,
        "MODULE 10 — FORECASTING REPORT",
        "=" * 60,
        "",
        f"Forecast horizon   : {FORECAST_YEAR}  ({horizon} years ahead)",
        f"ARIMA countries    : {len(arima_results)}",
        f"LSTM mode          : {lstm_mode}",
        f"Stationary series  : {n_stationary}/{len(adf_results)}",
        f"Mean backtest MAPE : {avg_mape:.1f}%"
                          if not np.isnan(avg_mape) else
        f"Mean backtest MAPE : N/A",
        f"Spec target MAPE   : < 15%",
        f"Spec met           : {'YES' if avg_mape < 15 else 'NO (more data recommended)'}",
        "",
        "ARIMA ORDERS SELECTED",
        "-" * 40,
    ]
    for country, r in arima_results.items():
        lines.append(f"  {country:20s}: ARIMA{r['order']}")

    if backtest_metrics:
        lines += ["", "BACKTEST ACCURACY", "-" * 40]
        for country, m in backtest_metrics.items():
            lines.append(f"  {country:20s}: MAPE={m['mape']:.1f}%  "
                         f"RMSE={m['rmse']:.0f}")

    lines += [
        "",
        "Figures saved: 10",
        f"Location: {FIG_DIR}",
        "",
        "=" * 60,
    ]

    path = REP_DIR / "module10_forecasting_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {path}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_STATE)
    logger.info("=" * 60)
    logger.info("MODULE 10 — FORECASTING")
    logger.info("=" * 60)

    # Load data
    agg           = load_time_series()
    global_series = build_global_series(agg)
    countries     = agg["country"].unique()

    # Stationarity
    logger.info("Running ADF stationarity tests...")
    adf_results = check_all_countries(agg)

    # ARIMA
    logger.info("Fitting ARIMA models...")
    arima_results, horizon = fit_all_arima(agg, countries)

    # ARIMA global forecast
    last_year       = int(global_series.index.max())
    global_horizon  = int(FORECAST_YEAR - last_year)
    global_arima, _ = fit_arima(global_series, "Global")
    arima_global_mean, arima_global_ci = arima_forecast(global_arima, global_horizon)
    future_years    = list(range(last_year + 1, FORECAST_YEAR + 1))
    arima_global_mean = pd.Series(arima_global_mean.values, index=future_years)

    # LSTM
    logger.info("Fitting LSTM...")
    model, hist_obj, scaler, lookback, scaled, lstm_mode = \
        fit_lstm_global(global_series)
    lstm_preds = lstm_forecast(model, scaler, lookback,
                               global_series.values if lstm_mode == "linear"
                               else scaled,
                               horizon, lstm_mode)

    # Backtest
    logger.info("Running backtests...")
    backtest_metrics = backtest_arima(agg, list(countries))

    # All 10 figures
    fig1_historical_trend(agg)
    fig2_adf_results(adf_results)
    fig3_arima_global(global_series, arima_results)
    fig4_arima_countries(arima_results)
    fig5_lstm_loss(hist_obj, lstm_mode)
    fig6_lstm_global(global_series, lstm_preds, lstm_mode)
    fig7_arima_vs_lstm(global_series, arima_global_mean, lstm_preds)
    fig8_confidence_intervals(arima_results)
    fig9_accuracy_table(backtest_metrics)
    fig10_forecast_heatmap(arima_results)

    # Save outputs
    fc_df  = save_forecasts(arima_results, global_series,
                             arima_global_mean, lstm_preds)
    report = write_report(adf_results, backtest_metrics,
                          arima_results, horizon, lstm_mode)

    logger.info("=" * 60)
    logger.info(f"MODULE 10 COMPLETE — 10 figures + 1 report")
    logger.info(f"Location: {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        logger.info(f"  {f.name}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  MODULE 10 COMPLETE")
    print("=" * 60)
    for line in report:
        print(" ", line)
    print(f"\n  Next: python scripts\\run_module11.py  (Dashboard)")


if __name__ == "__main__":
    main()
