"""
setup_database.py — SQLite Database Initialization
====================================================
Module 1 deliverable: creates the project database and all tables
that will be populated by subsequent modules.

Run once:  python scripts/setup_database.py

Why SQLite
----------
- Zero configuration, serverless, file-based
- Supports standard SQL
- Handles our expected data volume (< 1M rows) easily
- Ships with Python (no installation needed)
- Single .db file → easy to version-control and share
"""

import sys
from pathlib import Path

# Allow import of src utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import load_config, get_db_connection, setup_logging, write_module_summary


# ─────────────────────────────────────────────
# DDL — TABLE DEFINITIONS
# ─────────────────────────────────────────────
SCHEMA_SQL = """
-- --------------------------------------------------------
-- Raw World Bank indicators per country per year
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS world_bank_raw (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code        TEXT    NOT NULL,
    country_name        TEXT    NOT NULL,
    indicator_code      TEXT    NOT NULL,
    indicator_name      TEXT    NOT NULL,
    year                INTEGER NOT NULL,
    value               REAL,
    created_at          TEXT    DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Startup ecosystem data (from Kaggle / external sources)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS startup_ecosystem_raw (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    country             TEXT    NOT NULL,
    year                INTEGER NOT NULL,
    startup_count       INTEGER,
    total_funding_usd   REAL,
    venture_capital_usd REAL,
    industry            TEXT,
    created_at          TEXT    DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Cleaned / processed master dataset (output of Module 4)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS master_dataset (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    country                  TEXT    NOT NULL,
    country_code             TEXT,
    year                     INTEGER NOT NULL,

    -- Startup metrics
    startup_count            REAL,
    startup_funding_usd      REAL,
    venture_capital_usd      REAL,
    startup_count_growth_rate REAL,   -- TARGET VARIABLE

    -- Macroeconomic
    gdp_usd                  REAL,
    gdp_per_capita_usd       REAL,
    gdp_growth_rate          REAL,
    fdi_pct_gdp              REAL,

    -- Digital & innovation
    internet_penetration_pct REAL,
    research_expenditure_pct REAL,

    -- Labour market
    unemployment_rate        REAL,

    -- Education
    tertiary_enrollment_pct  REAL,

    -- Population
    population               REAL,

    -- Pandemic flag (1 if year >= 2020)
    pandemic_period          INTEGER,

    created_at               TEXT DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Engineered features (output of Module 5)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS engineered_features (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    country                 TEXT    NOT NULL,
    year                    INTEGER NOT NULL,
    startup_density         REAL,
    funding_per_capita      REAL,
    innovation_score        REAL,
    digital_readiness_score REAL,
    investment_efficiency   REAL,
    created_at              TEXT DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- ML model predictions
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    country         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    y_actual        REAL,
    y_predicted     REAL,
    residual        REAL,
    run_id          TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Forecasting output (LSTM)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    country         TEXT    NOT NULL,
    forecast_year   INTEGER NOT NULL,
    forecast_value  REAL,
    lower_bound     REAL,
    upper_bound     REAL,
    model_name      TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Clustering results (Module 12)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS cluster_assignments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    country         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    cluster_id      INTEGER,
    cluster_label   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Risk scores (Module 18)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    country         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    risk_score      REAL,
    stability_score REAL,
    confidence_score REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Experiment tracking (MLflow supplement)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT    NOT NULL,
    module          TEXT,
    model_name      TEXT,
    params          TEXT,   -- JSON string
    metrics         TEXT,   -- JSON string
    run_date        TEXT    DEFAULT (datetime('now'))
);

-- --------------------------------------------------------
-- Indexes for fast querying
-- --------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_wb_country_year  ON world_bank_raw(country_code, year);
CREATE INDEX IF NOT EXISTS idx_master_cy        ON master_dataset(country, year);
CREATE INDEX IF NOT EXISTS idx_preds_model      ON ml_predictions(model_name, country);
CREATE INDEX IF NOT EXISTS idx_forecast_country ON forecasts(country, forecast_year);
"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def setup_database():
    config = load_config()
    logger = setup_logging("setup_database", config)

    logger.info("Connecting to SQLite database...")
    conn = get_db_connection(config)
    cursor = conn.cursor()

    logger.info("Executing schema DDL...")
    cursor.executescript(SCHEMA_SQL)
    conn.commit()

    # Verify tables were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Tables created: {tables}")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;")
    indexes = [row[0] for row in cursor.fetchall()]
    logger.info(f"Indexes created: {indexes}")

    conn.close()

    # Write module summary
    summary = {
        "Status": "SUCCESS",
        "Tables created": len(tables),
        "Tables": ", ".join(tables),
        "Indexes created": len(indexes),
        "Database path": str(Path(config["paths"]["database"]))
    }
    report_path = write_module_summary("module_01_database", summary, config)
    logger.info(f"Summary written to: {report_path}")
    print(f"\n✓ Database initialized with {len(tables)} tables.")
    print(f"✓ Tables: {tables}")


if __name__ == "__main__":
    setup_database()
