"""
utils.py — Shared Utility Functions
====================================
Used across ALL modules in the Startup Growth Analytics System.
Handles: config loading, logging, seed setting, path resolution,
         saving figures, saving dataframes, and database connections.
"""

import os
import random
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PROJECT ROOT — resolves from any module location
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────
def load_config(config_path: str = None) -> dict:
    """
    Load the central config.yaml file.

    Returns
    -------
    dict : All project configuration parameters.

    Why needed
    ----------
    Centralising all parameters avoids magic numbers scattered across
    scripts. Any module can call load_config() to get paths, seeds,
    thresholds, and model hyperparameters in one place.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logging(module_name: str, config: dict = None) -> logging.Logger:
    """
    Configure and return a logger for a given module.

    Parameters
    ----------
    module_name : str  — e.g. 'data_collection', 'eda', 'ml'
    config      : dict — project config (loaded if None)

    Returns
    -------
    logging.Logger
    """
    if config is None:
        config = load_config()

    log_dir = PROJECT_ROOT / config["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "project.log"
    log_level = getattr(logging, config["logging"]["level"], logging.INFO)
    log_format = config["logging"]["format"]

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(module_name)
    return logger


# ─────────────────────────────────────────────
# REPRODUCIBILITY — SET ALL SEEDS
# ─────────────────────────────────────────────
def set_seeds(config: dict = None) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow.

    Why needed
    ----------
    Machine learning models (especially tree ensembles and neural networks)
    have stochastic components. Setting seeds ensures every run produces
    identical results — a core reproducibility requirement.

    Mathematical note
    -----------------
    Pseudo-random number generators (PRNGs) start from a seed value s₀
    and produce a deterministic sequence: s₁, s₂, ..., sₙ = f(sₙ₋₁).
    Same seed → same sequence → same model weights → same results.
    """
    if config is None:
        config = load_config()

    seed = config["reproducibility"]["random_seed"]
    random.seed(seed)
    np.random.seed(config["reproducibility"]["numpy_seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow seed (only if installed)
    try:
        import tensorflow as tf
        tf.random.set_seed(config["reproducibility"]["tensorflow_seed"])
    except ImportError:
        pass


# ─────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────
def get_path(key: str, config: dict = None) -> Path:
    """
    Resolve a named path from config relative to project root.

    Parameters
    ----------
    key : str — key in config['paths'], e.g. 'data_raw', 'figures'

    Returns
    -------
    Path (absolute)
    """
    if config is None:
        config = load_config()
    p = PROJECT_ROOT / config["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────
# FIGURE SAVING
# ─────────────────────────────────────────────
def save_figure(fig: plt.Figure, filename: str, config: dict = None, dpi: int = 150) -> Path:
    """
    Save a matplotlib figure to the outputs/figures directory.

    Parameters
    ----------
    fig      : plt.Figure
    filename : str  — e.g. 'eda_correlation_heatmap.png'
    config   : dict
    dpi      : int  — resolution (150 = good balance of quality/size)

    Returns
    -------
    Path to saved file.
    """
    if config is None:
        config = load_config()
    fig_dir = get_path("figures", config)
    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────
# DATAFRAME SAVE / LOAD
# ─────────────────────────────────────────────
def save_dataframe(df: pd.DataFrame, filename: str, stage: str = "interim", config: dict = None) -> Path:
    """
    Save a DataFrame as CSV to the appropriate data subfolder.

    Parameters
    ----------
    df       : pd.DataFrame
    filename : str   — e.g. 'world_bank_raw.csv'
    stage    : str   — 'raw' | 'interim' | 'processed' | 'external'
    config   : dict

    Returns
    -------
    Path to saved CSV.
    """
    if config is None:
        config = load_config()
    path_key = f"data_{stage}"
    out_dir = get_path(path_key, config)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


def load_dataframe(filename: str, stage: str = "interim", config: dict = None) -> pd.DataFrame:
    """
    Load a CSV from the appropriate data subfolder.

    Parameters
    ----------
    filename : str
    stage    : str  — 'raw' | 'interim' | 'processed' | 'external'

    Returns
    -------
    pd.DataFrame
    """
    if config is None:
        config = load_config()
    path_key = f"data_{stage}"
    file_path = get_path(path_key, config) / filename
    return pd.read_csv(file_path)


# ─────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────
def get_db_connection(config: dict = None) -> sqlite3.Connection:
    """
    Return a SQLite connection to the project database.

    Why SQLite
    ----------
    SQLite is serverless, requires no setup, stores data in a single file,
    and is sufficient for datasets of this scale (millions of rows).
    Every module can read/write structured data here without an external DB.

    Returns
    -------
    sqlite3.Connection
    """
    if config is None:
        config = load_config()
    db_path = PROJECT_ROOT / config["paths"]["database"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


# ─────────────────────────────────────────────
# MODULE REPORT GENERATION
# ─────────────────────────────────────────────
def write_module_summary(
    module_name: str,
    summary: dict,
    config: dict = None
) -> Path:
    """
    Write a plain-text summary report for a completed module.

    Parameters
    ----------
    module_name : str  — e.g. 'module_01_setup'
    summary     : dict — key-value pairs to record in the report

    Returns
    -------
    Path to report file.
    """
    if config is None:
        config = load_config()
    reports_dir = get_path("reports", config)
    out_path = reports_dir / f"{module_name}_summary.txt"

    lines = [
        "=" * 60,
        f"MODULE SUMMARY: {module_name.upper()}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        ""
    ]
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("=" * 60)

    out_path.write_text("\n".join(lines))
    return out_path


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    set_seeds(cfg)
    logger = setup_logging("utils_test", cfg)
    logger.info("Config loaded. Seed set. Logger active.")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Random seed: {cfg['reproducibility']['random_seed']}")
    print("utils.py — all checks passed.")
