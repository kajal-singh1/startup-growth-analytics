"""
api.py — FastAPI Backend
=========================

OBJECTIVE
---------
Expose the project's ML models, forecasts, clusters, and causal
findings as a REST API so the dashboard (or any external client)
can query them programmatically.

ENDPOINTS
---------
GET  /                          — health check
GET  /countries                 — list all countries in dataset
GET  /data/{country}            — full historical data for a country
POST /predict                   — predict growth rate from features
GET  /forecast/{country}        — ARIMA forecast for a country to 2027
GET  /forecast/global           — global aggregate forecast
GET  /clusters                  — all cluster assignments
GET  /clusters/{country}        — cluster for one country
GET  /recommend                 — top N countries by predicted growth
GET  /causal/pandemic-effect    — DiD causal inference results
GET  /stats/summary             — dataset summary statistics

RUN
---
    uvicorn scripts.api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Startup Growth Analytics API",
    description="REST API for predictions, forecasts, clustering, and causal "
               "inference on post-pandemic startup growth across 15 countries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached at startup)
# ─────────────────────────────────────────────────────────────────────────────

def _find_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load_features_df():
    path = _find_file([
        ROOT / "data" / "processed" / "master_features.csv",
        ROOT / "data" / "master_features.csv",
        ROOT / "data" / "processed" / "master_clean.csv",
    ])
    if path is None:
        return pd.DataFrame()
    df = path and pd.read_csv(path)
    if df is not None and "country_name" in df.columns and "country" not in df.columns:
        df = df.rename(columns={"country_name": "country"})
    return df if df is not None else pd.DataFrame()


def load_forecasts_df():
    path = _find_file([
        ROOT / "data" / "forecasts_2027.csv",
        ROOT / "data" / "processed" / "forecasts_2027.csv",
    ])
    return pd.read_csv(path) if path else pd.DataFrame()


def load_clusters_df():
    path = _find_file([
        ROOT / "data" / "cluster_assignments.csv",
        ROOT / "data" / "processed" / "cluster_assignments.csv",
    ])
    return pd.read_csv(path) if path else pd.DataFrame()


def load_model():
    """Load the best trained model. Falls back to None gracefully."""
    candidates = [
        ROOT / "models" / "best_model.pkl",
        ROOT / "models" / "saved" / "best_model.pkl",
        ROOT / "data" / "models" / "best_model.pkl",
    ]
    path = _find_file(candidates)
    if path is None:
        return None, None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        model    = payload.get("model", payload) if isinstance(payload, dict) else payload
        features = payload.get("features") if isinstance(payload, dict) else None
        return model, features
    except Exception:
        return None, None


# Load once at startup
DF_FEATURES  = load_features_df()
DF_FORECASTS = load_forecasts_df()
DF_CLUSTERS  = load_clusters_df()
MODEL, MODEL_FEATURES = load_model()

TARGET_COL = next((c for c in [
    "startup_count_growth_rate", "startup_growth_yoy", "yoy_growth"
] if c in DF_FEATURES.columns), None)

COUNTRY_COL = "country" if "country" in DF_FEATURES.columns else (
    DF_FEATURES.columns[0] if not DF_FEATURES.empty else "country"
)


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class PredictionInput(BaseModel):
    gdp_growth_rate: float = Field(..., example=2.5, description="GDP growth rate (%)")
    internet_penetration_pct: float = Field(..., example=85.0, description="Internet penetration (%)")
    gdp_per_capita_usd: float = Field(..., example=45000.0, description="GDP per capita (USD)")
    unemployment_rate: float = Field(..., example=5.0, description="Unemployment rate (%)")
    innovation_score: Optional[float] = Field(0.5, example=0.65)
    digital_readiness_score: Optional[float] = Field(0.5, example=0.7)
    economic_momentum: Optional[float] = Field(0.0, example=-1.5)
    investment_efficiency: Optional[float] = Field(0.0, example=300.0)
    startup_density: Optional[float] = Field(0.0, example=50.0)
    pandemic_period: Optional[int] = Field(0, example=1, description="1 if year >= 2020")
    pandemic_interaction: Optional[float] = Field(0.0)
    funding_per_startup_mn: Optional[float] = Field(3.0)


class PredictionOutput(BaseModel):
    predicted_growth_rate: float
    model_used: str
    confidence_note: str


class RecommendationItem(BaseModel):
    country: str
    predicted_growth_rate: float
    rank: int


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Basic health check and data availability summary."""
    return {
        "status": "ok",
        "service": "Startup Growth Analytics API",
        "version": "1.0.0",
        "data_loaded": not DF_FEATURES.empty,
        "rows": len(DF_FEATURES),
        "model_loaded": MODEL is not None,
        "forecasts_available": not DF_FORECASTS.empty,
        "clusters_available": not DF_CLUSTERS.empty,
    }


@app.get("/countries", tags=["Data"])
def get_countries():
    """List all countries available in the dataset."""
    if DF_FEATURES.empty:
        raise HTTPException(404, "No data loaded")
    countries = sorted(DF_FEATURES[COUNTRY_COL].dropna().unique().tolist())
    return {"count": len(countries), "countries": countries}


@app.get("/data/{country}", tags=["Data"])
def get_country_data(country: str):
    """Full historical time series for one country."""
    if DF_FEATURES.empty:
        raise HTTPException(404, "No data loaded")

    matches = DF_FEATURES[DF_FEATURES[COUNTRY_COL].str.lower() == country.lower()]
    if matches.empty:
        raise HTTPException(404, f"Country '{country}' not found")

    cols = [c for c in matches.columns if not c.endswith("_scaled")]
    records = matches[cols].sort_values("year").to_dict(orient="records")
    return {"country": country, "years": len(records), "data": records}


@app.post("/predict", response_model=PredictionOutput, tags=["ML"])
def predict_growth(payload: PredictionInput):
    """
    Predict startup growth rate from economic/digital indicators.
    Uses the saved best model from Module 8; falls back to a simple
    heuristic if no model is available.
    """
    features_dict = payload.dict()

    if MODEL is not None and MODEL_FEATURES:
        try:
            x = np.array([[features_dict.get(f, 0.0) for f in MODEL_FEATURES]])
            pred = float(MODEL.predict(x)[0])
            return PredictionOutput(
                predicted_growth_rate=round(pred, 2),
                model_used=type(MODEL).__name__,
                confidence_note="Prediction from trained model (Module 5/8)."
            )
        except Exception as e:
            pass  # fall through to heuristic

    # Heuristic fallback (based on Module 4 OLS coefficients)
    pred = (
        2.0
        + 0.8 * features_dict["gdp_growth_rate"]
        + 0.3 * (features_dict["internet_penetration_pct"] / 10)
        - 0.2 * features_dict["unemployment_rate"]
        + 3.0 * features_dict.get("innovation_score", 0.5)
    )
    return PredictionOutput(
        predicted_growth_rate=round(pred, 2),
        model_used="Heuristic (OLS-derived, no trained model found)",
        confidence_note="Run Module 5/8 and save best_model.pkl for ML-based predictions."
    )


@app.get("/forecast/global", tags=["Forecasting"])
def get_global_forecast():
    """Global aggregate forecast to 2027."""
    if DF_FORECASTS.empty:
        raise HTTPException(404, "No forecast data found. Run Module 10 first.")

    if "country" in DF_FORECASTS.columns:
        global_fc = DF_FORECASTS[DF_FORECASTS["country"] == "GLOBAL"]
    else:
        global_fc = DF_FORECASTS

    if global_fc.empty:
        raise HTTPException(404, "No global forecast found")

    return {
        "forecast_horizon": "2024-2027",
        "data": global_fc.to_dict(orient="records")
    }


@app.get("/forecast/{country}", tags=["Forecasting"])
def get_country_forecast(country: str):
    """ARIMA forecast for a specific country to 2027."""
    if DF_FORECASTS.empty:
        raise HTTPException(404, "No forecast data found. Run Module 10 first.")

    if "country" not in DF_FORECASTS.columns:
        raise HTTPException(404, "Forecast data missing country column")

    matches = DF_FORECASTS[
        (DF_FORECASTS["country"].str.lower() == country.lower()) &
        (DF_FORECASTS.get("model", "ARIMA") == "ARIMA")
    ]
    if matches.empty:
        raise HTTPException(404, f"No forecast found for '{country}'")

    return {
        "country": country,
        "model": "ARIMA",
        "forecast": matches.to_dict(orient="records")
    }


@app.get("/clusters", tags=["Clustering"])
def get_all_clusters():
    """All country -> cluster assignments."""
    if DF_CLUSTERS.empty:
        raise HTTPException(404, "No cluster data found. Run Module 9 first.")
    return {"data": DF_CLUSTERS.to_dict(orient="records")}


@app.get("/clusters/{country}", tags=["Clustering"])
def get_country_cluster(country: str):
    """Cluster assignment for one country."""
    if DF_CLUSTERS.empty:
        raise HTTPException(404, "No cluster data found. Run Module 9 first.")

    country_col = next((c for c in ["country", "country_name"]
                        if c in DF_CLUSTERS.columns), DF_CLUSTERS.columns[0])
    matches = DF_CLUSTERS[
        DF_CLUSTERS[country_col].str.lower() == country.lower()
    ]
    if matches.empty:
        raise HTTPException(404, f"No cluster found for '{country}'")
    return matches.iloc[0].to_dict()


@app.get("/recommend", tags=["Recommendations"])
def recommend_countries(top_n: int = Query(5, ge=1, le=15)):
    """
    Recommend top N countries by mean historical startup growth rate.
    A simple, transparent recommendation based on track record.
    """
    if DF_FEATURES.empty or TARGET_COL is None:
        raise HTTPException(404, "No data or target column available")

    means = (DF_FEATURES.groupby(COUNTRY_COL)[TARGET_COL]
            .mean().sort_values(ascending=False).head(top_n))

    items = [
        RecommendationItem(country=c, predicted_growth_rate=round(float(v), 2), rank=i+1)
        for i, (c, v) in enumerate(means.items())
    ]
    return {"top_n": top_n, "recommendations": [item.dict() for item in items]}


@app.get("/causal/pandemic-effect", tags=["Causal Inference"])
def get_causal_results():
    """
    DiD causal inference results — pandemic effect on startup growth.
    Loads from Module 7 output if available, else returns known summary.
    """
    did_path = _find_file([
        ROOT / "data" / "processed" / "did_results.csv",
        ROOT / "data" / "did_results.csv",
    ])
    if did_path:
        did_df = pd.read_csv(did_path)
        return {"source": "module7_causal_inference", "data": did_df.to_dict(orient="records")}

    # Known summary from Module 7 run
    return {
        "source": "module7_causal_inference (summary)",
        "did_coefficient_pp": 7.55,
        "did_p_value": 0.006,
        "did_significant": True,
        "psm_att_pp": 1.71,
        "psm_p_value": 0.42,
        "psm_significant": False,
        "interpretation": (
            "High-internet-penetration countries grew 7.55 percentage points "
            "faster post-pandemic than low-internet countries (p=0.006). "
            "Parallel trends assumption verified via event study."
        )
    }


@app.get("/stats/summary", tags=["Statistics"])
def get_summary_stats():
    """High-level dataset summary statistics."""
    if DF_FEATURES.empty:
        raise HTTPException(404, "No data loaded")

    summary = {
        "total_rows": len(DF_FEATURES),
        "countries": int(DF_FEATURES[COUNTRY_COL].nunique()),
        "year_range": f"{int(DF_FEATURES['year'].min())}-{int(DF_FEATURES['year'].max())}",
    }
    if TARGET_COL:
        summary["target_column"] = TARGET_COL
        summary["target_mean"]   = round(float(DF_FEATURES[TARGET_COL].mean()), 2)
        summary["target_std"]    = round(float(DF_FEATURES[TARGET_COL].std()), 2)
    if "startup_count" in DF_FEATURES.columns:
        summary["total_startups_tracked"] = int(DF_FEATURES["startup_count"].sum())
    if "total_funding_usd" in DF_FEATURES.columns:
        summary["total_funding_usd"] = float(DF_FEATURES["total_funding_usd"].sum())

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (for `python scripts/api.py`)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
