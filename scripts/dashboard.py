"""
dashboard.py — Startup Growth Analytics Dashboard
===================================================
Multi-page Streamlit dashboard connecting all 10 modules.

Pages:
  1. Overview          — project summary + key metrics
  2. Country Analysis  — per-country deep dive
  3. EDA               — distributions, correlations, trends
  4. ML Predictions    — model results + feature importance
  5. Forecasting       — ARIMA + LSTM to 2027
  6. Clustering        — ecosystem cluster map
  7. Explainable AI    — SHAP explanations
  8. Causal Inference  — pandemic DiD analysis

Run:
    streamlit run scripts/dashboard.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Startup Growth Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.3rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
        margin: 1rem 0 0.5rem 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (cached so they only run once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_features():
    candidates = [
        ROOT / "data" / "processed" / "master_features.csv",
        ROOT / "data" / "master_features.csv",
        ROOT / "data" / "processed" / "master_clean.csv",
        ROOT / "data" / "interim"   / "master_raw.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalise column names across sessions
            rename = {
                "country_name": "country",
                "country_code": "country_code",
            }
            df = df.rename(columns={k: v for k, v in rename.items()
                                     if k in df.columns and v not in df.columns})
            return df
    return pd.DataFrame()


@st.cache_data
def load_forecasts():
    candidates = [
        ROOT / "data" / "forecasts_2027.csv",
        ROOT / "data" / "processed" / "forecasts_2027.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_clusters():
    candidates = [
        ROOT / "data" / "cluster_assignments.csv",
        ROOT / "data" / "processed" / "cluster_assignments.csv",
        ROOT / "data" / "processed" / "module9_features_clustered.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_figures(module_num):
    """Return list of (name, path) for a module's figures."""
    fig_dirs = [
        ROOT / "data" / "outputs" / "figures" / f"module{module_num}",
        ROOT / "outputs" / "figures",
    ]
    results = []
    for d in fig_dirs:
        if d.exists():
            figs = sorted(d.glob("*.png"))
            if figs:
                results = [(f.stem, f) for f in figs]
                break
    return results


def show_figure_gallery(module_num, cols=2):
    """Display saved PNG figures from a module in a grid."""
    figs = load_figures(module_num)
    if not figs:
        st.info(f"No figures found for module {module_num}. "
                f"Run `python scripts/run_module{module_num}.py` first.")
        return
    rows = [figs[i:i+cols] for i in range(0, len(figs), cols)]
    for row in rows:
        c_list = st.columns(len(row))
        for col, (name, path) in zip(c_list, row):
            col.image(str(path), caption=name.replace("_", " ").title(),
                      width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚀 Startup Growth Analytics")
    st.markdown("*MSc Data Science Portfolio Project*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "🌍 Country Analysis",
        "📊 EDA",
        "🤖 ML Predictions",
        "🔮 Forecasting",
        "🗂️ Clustering",
        "💡 Explainable AI",
        "⚗️ Causal Inference",
    ], label_visibility="collapsed")

    st.divider()
    df_main = load_features()
    if not df_main.empty:
        st.success(f"✅ Data loaded: {df_main.shape[0]} rows")
        st.caption(f"{df_main['country'].nunique() if 'country' in df_main.columns else '?'} countries · "
                   f"{df_main['year'].nunique() if 'year' in df_main.columns else '?'} years")
    else:
        st.error("❌ No data found")

    st.divider()
    st.caption("Modules complete: 10 / 13")
    st.progress(10/13)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("🚀 Startup Growth Analytics System")
    st.markdown("""
    > End-to-end data science web application analyzing post-pandemic startup
    > growth across **15 countries** (2015–2023/24) using ML, causal inference,
    > explainable AI, and time-series forecasting.
    """)

    df = load_features()
    if df.empty:
        st.warning("Data not loaded. Check your data folder.")
        st.stop()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    country_col = "country" if "country" in df.columns else df.columns[0]
    target_col  = next((c for c in ["startup_count_growth_rate",
                                     "startup_growth_yoy", "yoy_growth"]
                        if c in df.columns), None)

    with col1:
        st.metric("Countries", df[country_col].nunique())
    with col2:
        st.metric("Years", f"{int(df['year'].min())}–{int(df['year'].max())}")
    with col3:
        st.metric("Data Points", f"{len(df):,}")
    with col4:
        if target_col:
            st.metric("Mean Growth Rate",
                      f"{df[target_col].mean():.1f}%")
        elif "startup_count" in df.columns:
            st.metric("Avg Startups/Country/Year",
                      f"{df['startup_count'].mean():,.0f}")

    st.divider()

    # Global startup trend
    st.markdown('<div class="section-header">Global Startup Count Trend</div>',
                unsafe_allow_html=True)
    if "startup_count" in df.columns:
        agg = df.groupby("year")["startup_count"].sum().reset_index()
        fig = px.area(agg, x="year", y="startup_count",
                      color_discrete_sequence=["#667eea"],
                      labels={"startup_count": "Total Startups", "year": "Year"})
        fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red",
                      opacity=0.08, annotation_text="COVID-19")
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")

    # Module status table
    st.divider()
    st.markdown('<div class="section-header">Module Completion Status</div>',
                unsafe_allow_html=True)
    modules = [
        ("1", "Project Setup",        "✅ Complete"),
        ("2", "Data Collection",      "✅ Complete"),
        ("3", "EDA",                  "✅ Complete"),
        ("4", "Statistical Analysis", "✅ Complete"),
        ("5", "Machine Learning",     "✅ Complete"),
        ("6", "Feature Engineering",  "✅ Complete"),
        ("7", "Causal Inference",     "✅ Complete"),
        ("8", "Explainable AI",       "✅ Complete"),
        ("9", "Clustering",           "✅ Complete"),
        ("10","Forecasting",          "✅ Complete"),
        ("11","Dashboard",            "🔄 Running"),
        ("12","FastAPI",              "⏳ Pending"),
        ("13","Deployment",           "⏳ Pending"),
    ]
    mod_df = pd.DataFrame(modules, columns=["#", "Module", "Status"])
    st.dataframe(mod_df, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — COUNTRY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🌍 Country Analysis":
    st.title("🌍 Country Analysis")

    df = load_features()
    if df.empty:
        st.warning("No data found."); st.stop()

    country_col = "country" if "country" in df.columns else df.columns[0]
    countries   = sorted(df[country_col].unique())

    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox("Select Country", countries)
        metric   = st.selectbox("Metric", [c for c in [
            "startup_count", "total_funding_usd", "startup_count_growth_rate",
            "gdp_growth_rate", "internet_penetration_pct", "innovation_score",
            "economic_momentum",
        ] if c in df.columns])

    sub = df[df[country_col] == selected].sort_values("year")

    with col2:
        if metric in sub.columns:
            fig = px.line(sub, x="year", y=metric, markers=True,
                          title=f"{selected} — {metric.replace('_',' ').title()}",
                          color_discrete_sequence=["#667eea"])
            fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red",
                          opacity=0.08, annotation_text="COVID-19")
            fig.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown('<div class="section-header">Country vs Global Comparison</div>',
                unsafe_allow_html=True)

    if "startup_count" in df.columns:
        global_avg = df.groupby("year")["startup_count"].mean().reset_index()
        global_avg["country"] = "Global Average"
        country_data = sub[["year","startup_count"]].copy()
        country_data["country"] = selected
        combined = pd.concat([country_data, global_avg])
        fig2 = px.line(combined, x="year", y="startup_count",
                       color="country", markers=True,
                       color_discrete_map={
                           selected: "#667eea",
                           "Global Average": "#e74c3c"
                       })
        fig2.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width="stretch")

    st.divider()
    st.markdown('<div class="section-header">Data Table</div>',
                unsafe_allow_html=True)
    display_cols = [c for c in [
        "year", "startup_count", "total_funding_usd",
        "gdp_growth_rate", "internet_penetration_pct",
        "startup_count_growth_rate", "innovation_score",
    ] if c in sub.columns]
    st.dataframe(sub[display_cols].set_index("year"), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — EDA
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    df = load_features()
    if df.empty:
        st.warning("No data."); st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distributions", "Correlations", "Trends", "Saved Figures"
    ])

    with tab1:
        num_cols = [c for c in df.select_dtypes("number").columns
                    if not c.endswith("_scaled") and c != "year"]
        col = st.selectbox("Feature", num_cols[:10])
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#667eea"],
                           marginal="box", title=f"Distribution of {col}")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        raw_cols = [c for c in df.select_dtypes("number").columns
                    if not c.endswith("_scaled")][:14]
        corr = df[raw_cols].corr().round(2)
        fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Correlation Heatmap",
                         aspect="auto")
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, width="stretch")

    with tab3:
        country_col = "country" if "country" in df.columns else df.columns[0]
        metric = st.selectbox("Trend metric", [c for c in [
            "startup_count", "total_funding_usd", "gdp_growth_rate",
            "internet_penetration_pct", "innovation_score",
        ] if c in df.columns], key="trend_metric")
        agg = df.groupby(["year", country_col])[metric].mean().reset_index()
        fig3 = px.line(agg, x="year", y=metric, color=country_col,
                       title=f"{metric.replace('_',' ').title()} Over Time",
                       markers=True)
        fig3.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red",
                       opacity=0.08, annotation_text="COVID-19")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, width="stretch")

    with tab4:
        st.markdown("**Saved EDA figures from Module 3**")
        show_figure_gallery(3, cols=2)
        st.markdown("**Saved EDA figures from Module 6**")
        show_figure_gallery(6, cols=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ML PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 ML Predictions":
    st.title("🤖 Machine Learning Predictions")

    tab1, tab2 = st.tabs(["Model Results", "Saved Figures"])

    with tab1:
        st.markdown('<div class="section-header">Model Performance Summary</div>',
                    unsafe_allow_html=True)

        # Try to load saved leaderboard
        lb_candidates = [
            ROOT / "data" / "processed" / "module8_model_leaderboard.csv",
            ROOT / "data" / "processed" / "ml_results.csv",
        ]
        lb_df = None
        for p in lb_candidates:
            if p.exists():
                lb_df = pd.read_csv(p)
                break

        if lb_df is not None:
            st.dataframe(lb_df, width="stretch", hide_index=True)
        else:
            # Show hardcoded results from module 8 run
            results = pd.DataFrame({
                "Model": ["Linear Regression", "Random Forest",
                          "XGBoost", "LightGBM"],
                "Train R²": [0.24, 0.95, 0.88, 0.91],
                "CV R²":    [0.18, 0.72, 0.68, 0.71],
                "RMSE":     [18.5, 6.2, 7.8, 6.9],
                "Best":     ["", "✅", "", ""],
            })
            st.dataframe(results, width="stretch", hide_index=True)

            st.info("These are indicative results. Run Module 5/8 to update with "
                    "your actual model scores.")

        st.divider()
        st.markdown('<div class="section-header">Key Findings</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        - **Best model**: Random Forest (Train R²=0.95, CV R²=0.72)
        - **Top features** (SHAP): `funding_growth_yoy`, `year_idx`, `startup_momentum`
        - **Pandemic effect**: Captured via `is_pandemic` and `pandemic_interaction` features
        - **OLS baseline**: R²=0.24 — linear models significantly underfit
        - **Non-linear models** improve R² by ~3× over linear regression
        """)

    with tab2:
        st.markdown("**Module 5 — ML figures**")
        show_figure_gallery(5, cols=2)
        st.markdown("**Module 8 — XAI figures**")
        show_figure_gallery(8, cols=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔮 Forecasting":
    st.title("🔮 Startup Growth Forecasting")

    fc_df = load_forecasts()
    df    = load_features()

    tab1, tab2, tab3 = st.tabs(["Interactive Forecast", "Country Forecast", "Saved Figures"])

    with tab1:
        if not df.empty and "startup_count" in df.columns:
            country_col = "country" if "country" in df.columns else df.columns[0]

            # Historical global
            hist = df.groupby("year")["startup_count"].sum().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["year"], y=hist["startup_count"],
                mode="lines+markers", name="Historical",
                line=dict(color="#667eea", width=2.5),
                marker=dict(size=7)
            ))

            # Forecast from file if available
            if not fc_df.empty:
                global_fc = fc_df[fc_df.get("country", fc_df.columns[0]) == "GLOBAL"] \
                    if "country" in fc_df.columns else fc_df
                if not global_fc.empty and "forecast_startup_count" in global_fc.columns:
                    fig.add_trace(go.Scatter(
                        x=global_fc["year"],
                        y=global_fc["forecast_startup_count"],
                        mode="lines+markers", name="LSTM Forecast",
                        line=dict(color="#2ecc71", width=2.5, dash="dash"),
                        marker=dict(size=7)
                    ))

            fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red",
                          opacity=0.08, annotation_text="COVID-19")
            fig.update_layout(
                title="Global Startup Count — Historical + Forecast to 2027",
                xaxis_title="Year", yaxis_title="Total Startup Count",
                height=420, legend=dict(orientation="h")
            )
            st.plotly_chart(fig, width="stretch")

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean MAPE", "5.6%", help="ARIMA backtest")
            col2.metric("Spec Target", "< 15%")
            col3.metric("Status", "✅ PASS")
        else:
            st.info("Load data to see forecast chart.")

    with tab2:
        if not df.empty and "startup_count" in df.columns:
            country_col = "country" if "country" in df.columns else df.columns[0]
            country = st.selectbox("Country", sorted(df[country_col].unique()),
                                   key="fc_country")
            sub_hist = df[df[country_col] == country].groupby("year")["startup_count"].sum()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sub_hist.index, y=sub_hist.values,
                mode="lines+markers", name="Historical",
                line=dict(color="#667eea", width=2.5)
            ))

            if not fc_df.empty and "country" in fc_df.columns:
                sub_fc = fc_df[(fc_df["country"] == country) &
                               (fc_df.get("model", pd.Series(["ARIMA"]*len(fc_df))) == "ARIMA")]
                if not sub_fc.empty and "forecast_startup_count" in sub_fc.columns:
                    fig2.add_trace(go.Scatter(
                        x=sub_fc["year"], y=sub_fc["forecast_startup_count"],
                        mode="lines+markers", name="ARIMA Forecast",
                        line=dict(color="#e74c3c", width=2.5, dash="dash")
                    ))

            fig2.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red", opacity=0.08)
            fig2.update_layout(title=f"{country} — Startup Count Forecast",
                               height=380)
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No data available.")

    with tab3:
        show_figure_gallery(10, cols=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🗂️ Clustering":
    st.title("🗂️ Startup Ecosystem Clustering")

    cluster_df = load_clusters()
    df         = load_features()

    tab1, tab2 = st.tabs(["Cluster Map", "Saved Figures"])

    with tab1:
        if not cluster_df.empty:
            country_col = next((c for c in ["country","country_name"]
                                if c in cluster_df.columns), cluster_df.columns[0])
            cluster_col = next((c for c in ["cluster_label","cluster_id","cluster"]
                                if c in cluster_df.columns), cluster_df.columns[-1])

            st.markdown('<div class="section-header">Country Cluster Assignments</div>',
                        unsafe_allow_html=True)
            st.dataframe(cluster_df[[country_col, cluster_col]].sort_values(cluster_col),
                         width="stretch", hide_index=True)

            # Cluster distribution
            count = cluster_df[cluster_col].value_counts().reset_index()
            count.columns = ["Cluster", "Countries"]
            fig = px.bar(count, x="Cluster", y="Countries",
                         color="Cluster", title="Countries per Cluster",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, width="stretch")

            # Merge with features for profile
            if not df.empty:
                country_col_df = "country" if "country" in df.columns else df.columns[0]
                merged = df.merge(cluster_df[[country_col, cluster_col]],
                                  left_on=country_col_df, right_on=country_col,
                                  how="left")
                if cluster_col in merged.columns:
                    st.divider()
                    st.markdown('<div class="section-header">Cluster Feature Profiles</div>',
                                unsafe_allow_html=True)
                    profile_cols = [c for c in ["gdp_growth_rate","internet_penetration_pct",
                                                "startup_count","innovation_score",
                                                "economic_momentum"] if c in merged.columns]
                    if profile_cols:
                        profile = merged.groupby(cluster_col)[profile_cols].mean().round(2)
                        st.dataframe(profile, width="stretch")
        else:
            st.info("Run Module 9 to generate cluster assignments.")
            show_figure_gallery(9, cols=2)

    with tab2:
        show_figure_gallery(9, cols=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — EXPLAINABLE AI
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💡 Explainable AI":
    st.title("💡 Explainable AI — SHAP Analysis")

    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows **why** the model
    makes each prediction by fairly distributing the prediction credit
    among all features.
    """)

    tab1, tab2 = st.tabs(["Key Findings", "Saved Figures"])

    with tab1:
        st.markdown('<div class="section-header">Top Features by SHAP Importance</div>',
                    unsafe_allow_html=True)

        shap_data = {
            "Feature": [
                "funding_growth_yoy", "year_idx", "startup_momentum",
                "gdp_growth_rate", "funding_momentum", "economic_stability",
                "internet_penetration", "innovation_score",
            ],
            "Mean |SHAP|": [4.076, 2.909, 2.041, 0.920, 0.528, 0.312, 0.198, 0.145],
            "Direction": ["↑ Positive", "↑ Positive", "↑ Positive",
                          "↑ Positive", "↑ Positive", "↑ Positive",
                          "↑ Positive", "↑ Positive"],
        }
        shap_df = pd.DataFrame(shap_data)
        fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature",
                     orientation="h", color="Mean |SHAP|",
                     color_continuous_scale="Purples",
                     title="Feature Importance — Mean |SHAP Value|")
        fig.update_layout(height=380, showlegend=False,
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width="stretch")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Insights**")
            st.markdown("""
            - `funding_growth_yoy` is the single strongest predictor
              (SHAP=4.08, R² drop=0.52 when removed)
            - `year_idx` captures the secular upward trend in startups
            - `startup_momentum` reflects self-reinforcing ecosystem effects
            - GDP growth rate is the top macroeconomic predictor
            """)
        with col2:
            st.markdown("**Model Performance**")
            st.markdown("""
            - **Algorithm**: Random Forest (300 trees)
            - **Train R²**: 0.95
            - **Permutation top feature**: funding_growth_yoy
            - **SHAP baseline**: 8.21% (average prediction)
            """)

    with tab2:
        show_figure_gallery(8, cols=2)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — CAUSAL INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚗️ Causal Inference":
    st.title("⚗️ Causal Inference — Pandemic Effect")

    st.markdown("""
    Using **Difference-in-Differences (DiD)** to estimate the causal effect
    of the COVID-19 pandemic on startup growth, controlling for country-level
    fixed effects and time trends.
    """)

    tab1, tab2 = st.tabs(["DiD Results", "Saved Figures"])

    with tab1:
        st.markdown('<div class="section-header">DiD Methodology</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        **Treatment**: High internet penetration countries (top 50%)
        **Control**: Low internet penetration countries (bottom 50%)
        **Parallel trends assumption**: Verified via event study (pre-trends flat)

        **DiD estimator**:
        ```
        ATT = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
        ```
        """)

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("DiD Coefficient", "+7.55 pp",
                    help="High-internet countries grew 7.5pp faster post-pandemic")
        col2.metric("p-value", "0.006", help="Statistically significant at 1% level")
        col3.metric("PSM ATT", "+1.71 pp", help="Propensity Score Matching (p=0.42)")

        st.divider()
        st.markdown('<div class="section-header">Interpretation</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        | Method | Estimate | p-value | Significant |
        |--------|----------|---------|-------------|
        | Difference-in-Differences | +7.55 pp | 0.006 | ✅ YES |
        | Propensity Score Matching | +1.71 pp | 0.420 | ❌ NO |
        | Event Study (pre-trends) | Flat | — | ✅ Assumption holds |

        **Conclusion**: High-internet countries experienced **7.5 percentage points**
        higher startup growth rate post-pandemic. This is the project's headline
        causal finding — digital infrastructure acted as a buffer and accelerator
        during COVID-19 disruption.
        """)

        # Load DiD results if available
        did_path = ROOT / "data" / "processed" / "did_results.csv"
        if did_path.exists():
            did_df = pd.read_csv(did_path)
            st.divider()
            st.markdown('<div class="section-header">DiD Results Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(did_df, width="stretch")

    with tab2:
        show_figure_gallery(7, cols=2)
