"""Shared utility functions for Startup Growth Analytics System."""

import os
import yaml
import logging
import sqlite3
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    cfg = load_config()
    log_dir = PROJECT_ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d}.log"

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ── Seeds ─────────────────────────────────────────────────────────────────────
def set_seeds(seed: int = None):
    if seed is None:
        seed = load_config()["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ── Database ──────────────────────────────────────────────────────────────────
def get_db_connection() -> sqlite3.Connection:
    cfg = load_config()
    db_path = PROJECT_ROOT / cfg["paths"]["database"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ── DataFrames ────────────────────────────────────────────────────────────────
def save_dataframe(df: pd.DataFrame, relative_path: str, index: bool = False):
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def load_dataframe(relative_path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path, **kwargs)
