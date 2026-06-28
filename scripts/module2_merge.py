"""
Module 2 – Step 3: Merge & Master Dataset
Merges World Bank + Startup Ecosystem data into a single master CSV.
Performs validation, adds derived features, saves to DB and CSV.
"""

import pandas as pd
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, get_db_connection, get_project_root
from scripts.module2_world_bank import load_world_bank_data
from scripts.module2_startup_data import load_startup_data

logger = get_logger("merge_data")


def merge_datasets() -> pd.DataFrame:
    wb = load_world_bank_data()
    se = load_startup_data()

    # Drop duplicate pandemic_period column before merge (we'll recompute)
    wb = wb.drop(columns=["pandemic_period"], errors="ignore")
    se = se.drop(columns=["pandemic_period"], errors="ignore")

    master = pd.merge(
        wb, se,
        on=["country_code", "country_name", "year"],
        how="inner"
    )

    # Pandemic period labels
    master["pandemic_period"] = master["year"].apply(
        lambda y: "pre" if y < 2020 else ("during" if y <= 2021 else "post")
    )

    # Partial data flag — 2024 is H1 annualised estimates
    master["is_partial"] = (master["year"] == 2024).astype(int)

    # ── Derived Features ───────────────────────────────────────────────────────
    # Startup density (startups per million population)
    master["startup_density"] = (master["startup_count"] / master["population_mn"]).round(2)

    # Funding intensity (funding USD mn per billion GDP)
    master["funding_intensity"] = (master["total_funding_usd_mn"] / master["gdp_usd_bn"]).round(4)

    # Unicorn production rate (unicorns per 1000 startups)
    master["unicorn_rate"] = ((master["num_unicorns"] / master["startup_count"]) * 1000).round(4)

    # Startup growth YoY (% change in startup count)
    master = master.sort_values(["country_code", "year"])
    master["startup_count_lag1"] = master.groupby("country_code")["startup_count"].shift(1)
    master["startup_growth_yoy"] = (
        (master["startup_count"] - master["startup_count_lag1"]) / master["startup_count_lag1"] * 100
    ).round(2)
    master = master.drop(columns=["startup_count_lag1"])

    # Funding growth YoY
    master["funding_lag1"] = master.groupby("country_code")["total_funding_usd_mn"].shift(1)
    master["funding_growth_yoy"] = (
        (master["total_funding_usd_mn"] - master["funding_lag1"]) / master["funding_lag1"] * 100
    ).round(2)
    master = master.drop(columns=["funding_lag1"])

    # Is pandemic year flag
    master["is_pandemic"] = master["year"].isin([2020, 2021]).astype(int)

    # Is post-pandemic flag (includes 2024 partial)
    master["is_post_pandemic"] = master["year"].isin([2022, 2023, 2024]).astype(int)

    return master


def validate(df: pd.DataFrame) -> dict:
    report = {
        "total_rows": len(df),
        "countries": df["country_code"].nunique(),
        "years": sorted(df["year"].unique().tolist()),
        "missing_values": df.isnull().sum().to_dict(),
        "shape": df.shape,
    }

    # Coverage by period
    report["period_counts"] = df["pandemic_period"].value_counts().to_dict()

    # Key stat ranges
    for col in ["startup_count", "total_funding_usd_mn", "gdp_growth_rate", "internet_penetration"]:
        report[f"{col}_range"] = (round(df[col].min(), 2), round(df[col].max(), 2))

    return report


def run():
    logger.info("=== Module 2 Step 3: Merge & Validate ===")

    master = merge_datasets()
    logger.info(f"Master dataset shape: {master.shape}")

    # Save CSV
    out_path = get_project_root() / "data/processed/master_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)
    logger.info("Saved → data/processed/master_dataset.csv")

    # Save to DB
    conn = get_db_connection()
    master.to_sql("master_dataset", conn, if_exists="replace", index=False)
    conn.close()
    logger.info("Saved → SQLite: master_dataset table")

    # Validate
    report = validate(master)
    return master, report


if __name__ == "__main__":
    master, report = run()
    print(f"\n✓ Master dataset: {report['shape']}")
    print(f"  Countries  : {report['countries']}")
    print(f"  Years      : {report['years']}")
    print(f"  Period dist: {report['period_counts']}")
    print(f"  Missing values (non-null expected): {sum(v for v in report['missing_values'].values() if v > 0)} cells")

    print("\n── Key Ranges ──────────────────────────────────────")
    for k, v in report.items():
        if k.endswith("_range"):
            print(f"  {k:<40} {v}")
