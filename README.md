# Startup Growth Analytics System

> End-to-end, fully open-source data science application for analyzing, predicting, explaining, and forecasting post-pandemic startup growth across countries.

---

## Project Overview

This system applies statistics, causal inference, machine learning, deep learning, explainable AI, and interactive dashboards to real-world startup ecosystem data. Every library, dataset, and deployment platform used is free and open-source.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/startup_growth_analytics.git
cd startup_growth_analytics

# 2. Create environment (choose one)
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate startup_growth_analytics

# 3. Initialize database
python scripts/setup_database.py

# 4. Verify setup
python scripts/verify_setup.py
```

---

## Project Structure

```
startup_growth_analytics/
│
├── config/                  # Central configuration
│   └── config.yaml
│
├── data/
│   ├── raw/                 # Original downloaded datasets (never modified)
│   ├── interim/             # Partially processed data
│   ├── processed/           # Final clean datasets used for modelling
│   └── external/            # Third-party reference data
│
├── database/
│   └── startup_growth.db    # SQLite database
│
├── docs/                    # Documentation and research notes
│
├── models/
│   ├── saved/               # Trained model files
│   ├── tuned/               # Hyperparameter-optimized models
│   └── experiments/         # MLflow experiment artifacts
│
├── notebooks/               # Jupyter notebooks (exploration)
│
├── outputs/
│   ├── figures/             # All saved plots
│   ├── reports/             # Module summary reports
│   └── logs/                # Application logs + MLflow runs
│
├── scripts/                 # Runnable scripts (setup, verify, etc.)
│
├── src/                     # Source code — one subfolder per module
│   ├── utils.py             # Shared utility functions
│   ├── collection/          # Module 2: Data Collection
│   ├── validation/          # Module 3: Data Validation
│   ├── cleaning/            # Module 4: Data Cleaning
│   ├── features/            # Module 5: Feature Engineering
│   ├── eda/                 # Module 6: Exploratory Data Analysis
│   ├── stats/               # Module 7: Statistical Analysis
│   ├── causal/              # Module 8: Causal Inference
│   ├── ml/                  # Module 9: Machine Learning
│   ├── xai/                 # Module 11: Explainable AI
│   ├── clustering/          # Module 12: Clustering
│   ├── forecasting/         # Module 16: LSTM Forecasting
│   ├── geospatial/          # Module 17: Geospatial Analytics
│   ├── risk/                # Module 18: Risk Assessment
│   ├── api/                 # Module 19: FastAPI
│   ├── dashboard/           # Module 20: Streamlit
│   ├── reports/             # Module 21: Report Generation
│   └── mlops/               # Module 22: MLflow + DVC
│
├── tests/                   # Unit tests
│
├── requirements.txt
└── environment.yml
```

---

## Modules

| # | Module | Status |
|---|--------|--------|
| 1 | Project Setup | ✅ Complete |
| 2 | Data Collection | 🔜 Next |
| 3 | Data Validation | ⏳ Pending |
| 4 | Data Cleaning | ⏳ Pending |
| 5 | Feature Engineering | ⏳ Pending |
| 6 | Exploratory Data Analysis | ⏳ Pending |
| 7 | Statistical Analysis | ⏳ Pending |
| 8 | Causal Inference | ⏳ Pending |
| 9 | Machine Learning | ⏳ Pending |
| 10 | Hyperparameter Tuning | ⏳ Pending |
| 11 | Explainable AI | ⏳ Pending |
| 12 | Clustering | ⏳ Pending |
| 13 | Anomaly Detection | ⏳ Pending |
| 14 | Startup Success Prediction | ⏳ Pending |
| 15 | Recommendation System | ⏳ Pending |
| 16 | Forecasting (LSTM) | ⏳ Pending |
| 17 | Geospatial Analytics | ⏳ Pending |
| 18 | Risk Assessment | ⏳ Pending |
| 19 | API Development (FastAPI) | ⏳ Pending |
| 20 | Dashboard (Streamlit) | ⏳ Pending |
| 21 | Report Generation | ⏳ Pending |
| 22 | MLOps | ⏳ Pending |
| 23 | Deployment | ⏳ Pending |

---

## Technology Stack

- **Language:** Python 3.12+
- **Data:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML:** Scikit-learn, XGBoost, LightGBM
- **Deep Learning:** TensorFlow / Keras
- **Explainable AI:** SHAP
- **Causal Inference:** DoWhy, EconML
- **Geospatial:** GeoPandas, Folium
- **Dashboard:** Streamlit
- **API:** FastAPI
- **Database:** SQLite
- **MLOps:** MLflow, DVC

---

## Reproducibility

All random seeds are set in `config/config.yaml`. Every module reads seeds from config before running any stochastic operation.

```python
from src.utils import set_seeds, load_config
cfg = load_config()
set_seeds(cfg)  # Sets Python, NumPy, TensorFlow seeds
```

---

## Data Sources (Free & Open)

- [World Bank Open Data](https://data.worldbank.org)
- [OECD Data](https://data.oecd.org)
- [Our World in Data](https://ourworldindata.org)
- [Kaggle Public Datasets](https://www.kaggle.com/datasets)
- [Global Innovation Index](https://www.wipo.int/global_innovation_index)

---

## License

MIT License — free to use, modify, and distribute.
