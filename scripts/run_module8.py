"""
run_module8.py — Module 8: Explainable AI
==========================================
Usage:
    python scripts\run_module8.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.module8_xai import main

if __name__ == "__main__":
    main()
