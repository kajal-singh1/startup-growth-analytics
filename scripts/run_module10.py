"""
run_module10.py — Module 10: Forecasting
==========================================
Usage:
    python scripts\run_module10.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.module10_forecasting import main

if __name__ == "__main__":
    main()
