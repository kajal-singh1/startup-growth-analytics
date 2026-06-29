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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(module_name: str, config: dict = None):
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
    return logging.getLogger(module_name)


def set_seeds(config: dict = None):
    if config is None:
        config = load_config()
    seed = config["reproducibility"]["random_seed"]
    random.seed(seed)
    np.random.seed(config["reproducibility"]["numpy_seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(config["reproducibility"]["tensorflow_seed"])
    except ImportError:
        pass


def get_path(key: str, config: dict = None) -> Path:
    if config is None:
        config = load_config()
    p = PROJECT_ROOT / config["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, filename: str, config: dict = None, dpi: int = 150) -> Path:
    if config is None:
        config = load_config()
    fig_dir = get_path("figures", config)
    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_dataframe(df: pd.DataFrame, filename: str, stage: str = "interim", config: dict = None) -> Path:
    if config is None:
        config = load_config()
    out_dir = get_path(f"data_{stage}", config)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


def load_dataframe(filename: str, stage: str = "interim", config: dict = None) -> pd.DataFrame:
    if config is None:
        config = load_config()
    file_path = get_path(f"data_{stage}", config) / filename
    return pd.read_csv(file_path)


def get_db_connection(config: dict = None) -> sqlite3.Connection:
    if config is None:
        config = load_config()
    db_path = PROJECT_ROOT / config["paths"]["database"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def write_module_summary(module_name: str, summary: dict, config: dict = None) -> Path:
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

# ── Aliases for module compatibility ──────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    return setup_logging(name)


def get_project_root() -> Path:
    return PROJECT_ROOT