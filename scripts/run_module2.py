"""
Module 2 – Main Runner
Runs all 3 data collection steps in sequence and produces a validation report.

Usage:
    python scripts/run_module2.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, get_project_root

logger = get_logger("module2_runner")


def run():
    logger.info("=" * 60)
    logger.info("MODULE 2: DATA COLLECTION — START")
    logger.info("=" * 60)

    results = {}

    # Step 1 — World Bank data
    print("\n[1/3] Loading World Bank data...")
    from scripts.module2_world_bank import run as run_wb
    wb_df = run_wb()
    results["world_bank"] = {
        "rows": len(wb_df),
        "countries": wb_df["country_code"].nunique(),
        "years": sorted(wb_df["year"].unique().tolist()),
        "columns": wb_df.columns.tolist()
    }
    print(f"  ✓ {len(wb_df)} rows | {wb_df['country_code'].nunique()} countries | {len(wb_df.columns)} columns")

    # Step 2 — Startup ecosystem data
    print("\n[2/3] Loading startup ecosystem data...")
    from scripts.module2_startup_data import run as run_se
    se_df = run_se()
    results["startup_ecosystem"] = {
        "rows": len(se_df),
        "countries": se_df["country_code"].nunique(),
        "top_sectors": se_df["top_sector"].value_counts().head(5).to_dict()
    }
    print(f"  ✓ {len(se_df)} rows | {se_df['country_code'].nunique()} countries | {len(se_df.columns)} columns")

    # Step 3 — Merge
    print("\n[3/3] Merging into master dataset...")
    from scripts.module2_merge import run as run_merge
    master, validation = run_merge()
    results["master"] = {
        "shape": list(validation["shape"]),
        "missing_values_total": sum(v for v in validation["missing_values"].values() if v > 0),
        "period_distribution": validation["period_counts"],
    }
    print(f"  ✓ Master: {validation['shape']} | Missing cells: {results['master']['missing_values_total']}")

    # Validation report
    print("\n" + "=" * 60)
    print("MODULE 2 VALIDATION REPORT")
    print("=" * 60)

    master_period = master.groupby("pandemic_period").agg(
        avg_startup_count=("startup_count", "mean"),
        avg_funding_mn=("total_funding_usd_mn", "mean"),
        avg_gdp_growth=("gdp_growth_rate", "mean"),
        avg_internet_pct=("internet_penetration", "mean"),
        total_unicorns=("num_unicorns", "sum"),
    ).round(2)

    # Reorder
    order = ["pre", "during", "post"]
    master_period = master_period.reindex([o for o in order if o in master_period.index])

    print("\n📊 Average metrics by pandemic period (all countries):")
    print(master_period.to_string())

    print("\n🏆 Top 5 countries by total 2023 funding:")
    top5 = master[master["year"] == 2023].nlargest(5, "total_funding_usd_mn")[
        ["country_name", "startup_count", "total_funding_usd_mn", "num_unicorns"]
    ]
    print(top5.to_string(index=False))

    print("\n🌐 Internet penetration vs startup density (2023):")
    inet_table = master[master["year"] == 2023][
        ["country_name", "internet_penetration", "startup_density", "funding_intensity"]
    ].sort_values("startup_density", ascending=False)
    print(inet_table.to_string(index=False))

    # Write report file
    report_path = get_project_root() / "reports" / "module2_validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"MODULE 2 VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("DATASET SUMMARY\n")
        f.write(f"  World Bank rows   : {results['world_bank']['rows']}\n")
        f.write(f"  Startup Eco rows  : {results['startup_ecosystem']['rows']}\n")
        f.write(f"  Master shape      : {results['master']['shape']}\n")
        f.write(f"  Missing cells     : {results['master']['missing_values_total']}\n")
        f.write(f"  Period dist.      : {results['master']['period_distribution']}\n\n")
        f.write("PERIOD ANALYSIS\n")
        f.write(master_period.to_string() + "\n\n")
        f.write("TOP 5 BY 2023 FUNDING\n")
        f.write(top5.to_string(index=False) + "\n\n")
        f.write("INTERNET VS STARTUP DENSITY (2023)\n")
        f.write(inet_table.to_string(index=False) + "\n")

    print(f"\n📄 Report saved → reports/module2_validation_report.txt")

    print("\n" + "=" * 60)
    print("✓ MODULE 2 COMPLETE — All 3 steps passed")
    print("✓ Data ready in: data/raw/ and data/processed/")
    print("✓ SQLite tables: world_bank_raw, startup_ecosystem_raw, master_dataset")
    print("=" * 60)
    print("\nNext: run `python scripts/run_module3.py` for EDA")

    return master


if __name__ == "__main__":
    run()
