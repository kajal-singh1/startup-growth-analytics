"""
run_module9.py — Module 9: Clustering
======================================
Usage:
    python scripts\run_module9.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.module9_clustering import main

if __name__ == "__main__":
    main()
