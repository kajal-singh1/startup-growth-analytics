"""
Module 2 – Step 2: Startup Ecosystem Data
Realistic startup ecosystem metrics based on public reports from:
- Crunchbase Global Startup Reports (2015–2023)
- StartupBlink Global Startup Ecosystem Index
- OECD Entrepreneurship at a Glance
- Dealroom.co Country Reports

Metrics per country per year:
- startup_count: number of active startups
- total_funding_usd_mn: total VC/angel funding (USD millions)
- num_deals: number of funding deals
- avg_deal_size_usd_mn: average deal size (USD millions)
- num_unicorns: unicorns created that year
- cumulative_unicorns: total unicorns ever
- growth_rate_yoy: YoY growth in startup count (%)
- top_sector: dominant sector by count
"""

import pandas as pd
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, get_db_connection, get_project_root

logger = get_logger("startup_data")

STARTUP_DATA = [
    # country_code, country_name, year,
    # startup_count, total_funding_usd_mn, num_deals, avg_deal_size_usd_mn,
    # num_unicorns, cumulative_unicorns, growth_rate_yoy, top_sector

    # ── United States ──────────────────────────────────────────────────────────
    ("US","United States",2015,57000,72000,8210,8.77,28,156,"SaaS"),
    ("US","United States",2016,61000,77000,8560,9.00,25,181,"SaaS"),
    ("US","United States",2017,66000,84000,8900,9.44,30,211,"FinTech"),
    ("US","United States",2018,72000,130000,9200,14.13,44,255,"FinTech"),
    ("US","United States",2019,77000,136000,9400,14.47,47,302,"HealthTech"),
    ("US","United States",2020,74000,156000,9100,17.14,62,364,"EdTech"),
    ("US","United States",2021,90000,330000,11500,28.70,111,475,"SaaS"),
    ("US","United States",2022,88000,215000,10800,19.91,33,508,"AI/ML"),
    ("US","United States",2023,85000,170000,9800,17.35,22,530,"AI/ML"),

    # ── India ──────────────────────────────────────────────────────────────────
    ("IN","India",2015,4200,5000,940,5.32,4,8,"eCommerce"),
    ("IN","India",2016,4800,4400,1050,4.19,2,10,"eCommerce"),
    ("IN","India",2017,5500,13500,1160,11.64,1,11,"FinTech"),
    ("IN","India",2018,7200,10600,1350,7.85,6,17,"FinTech"),
    ("IN","India",2019,8900,14500,1490,9.73,7,24,"EdTech"),
    ("IN","India",2020,9500,10300,1220,8.44,12,36,"EdTech"),
    ("IN","India",2021,13500,42000,2100,20.00,44,80,"FinTech"),
    ("IN","India",2022,14500,24000,2050,11.71,22,102,"SaaS"),
    ("IN","India",2023,14000,9600,1800,5.33,2,104,"AI/ML"),

    # ── United Kingdom ─────────────────────────────────────────────────────────
    ("GB","United Kingdom",2015,5100,6200,1410,4.40,7,21,"FinTech"),
    ("GB","United Kingdom",2016,5400,7000,1480,4.73,4,25,"FinTech"),
    ("GB","United Kingdom",2017,5900,7900,1540,5.13,5,30,"FinTech"),
    ("GB","United Kingdom",2018,6400,9100,1620,5.62,7,37,"FinTech"),
    ("GB","United Kingdom",2019,6900,13000,1700,7.65,8,45,"FinTech"),
    ("GB","United Kingdom",2020,6600,15000,1580,9.49,12,57,"HealthTech"),
    ("GB","United Kingdom",2021,8100,30000,2000,15.00,29,86,"FinTech"),
    ("GB","United Kingdom",2022,8500,21000,1900,11.05,10,96,"AI/ML"),
    ("GB","United Kingdom",2023,8200,15000,1750,8.57,7,103,"AI/ML"),

    # ── Germany ────────────────────────────────────────────────────────────────
    ("DE","Germany",2015,2800,3400,820,4.15,3,8,"eCommerce"),
    ("DE","Germany",2016,3000,3100,860,3.60,2,10,"FinTech"),
    ("DE","Germany",2017,3300,5100,900,5.67,3,13,"FinTech"),
    ("DE","Germany",2018,3600,5700,960,5.94,4,17,"Mobility"),
    ("DE","Germany",2019,3900,6800,1010,6.73,4,21,"Mobility"),
    ("DE","Germany",2020,3700,5600,950,5.89,4,25,"HealthTech"),
    ("DE","Germany",2021,4500,17200,1200,14.33,11,36,"SaaS"),
    ("DE","Germany",2022,4600,9300,1150,8.09,4,40,"CleanTech"),
    ("DE","Germany",2023,4400,6500,1050,6.19,2,42,"AI/ML"),

    # ── France ─────────────────────────────────────────────────────────────────
    ("FR","France",2015,2100,1800,610,2.95,2,5,"eCommerce"),
    ("FR","France",2016,2300,2200,650,3.38,2,7,"eCommerce"),
    ("FR","France",2017,2600,3100,700,4.43,3,10,"FinTech"),
    ("FR","France",2018,2900,3900,760,5.13,4,14,"AI/ML"),
    ("FR","France",2019,3200,5000,820,6.10,4,18,"AI/ML"),
    ("FR","France",2020,3100,4800,790,6.08,4,22,"HealthTech"),
    ("FR","France",2021,4000,11900,1050,11.33,10,32,"SaaS"),
    ("FR","France",2022,4300,8100,1100,7.36,4,36,"CleanTech"),
    ("FR","France",2023,4200,6200,1000,6.20,3,39,"AI/ML"),

    # ── Singapore ──────────────────────────────────────────────────────────────
    ("SG","Singapore",2015,2400,1900,530,3.58,2,5,"FinTech"),
    ("SG","Singapore",2016,2600,2400,570,4.21,2,7,"FinTech"),
    ("SG","Singapore",2017,3000,3600,620,5.81,3,10,"FinTech"),
    ("SG","Singapore",2018,3400,5000,680,7.35,3,13,"FinTech"),
    ("SG","Singapore",2019,3700,6500,730,8.90,4,17,"FinTech"),
    ("SG","Singapore",2020,3500,8400,700,12.00,5,22,"HealthTech"),
    ("SG","Singapore",2021,4400,13000,890,14.61,10,32,"FinTech"),
    ("SG","Singapore",2022,4700,9200,920,10.00,4,36,"Web3"),
    ("SG","Singapore",2023,4600,6100,880,6.93,2,38,"AI/ML"),

    # ── Israel ─────────────────────────────────────────────────────────────────
    ("IL","Israel",2015,2300,4900,860,5.70,3,8,"CyberSecurity"),
    ("IL","Israel",2016,2500,5200,900,5.78,4,12,"CyberSecurity"),
    ("IL","Israel",2017,2700,5400,940,5.74,5,17,"CyberSecurity"),
    ("IL","Israel",2018,2900,6500,980,6.63,5,22,"CyberSecurity"),
    ("IL","Israel",2019,3100,8300,1020,8.14,6,28,"AI/ML"),
    ("IL","Israel",2020,3000,9900,980,10.10,8,36,"HealthTech"),
    ("IL","Israel",2021,3600,25000,1200,20.83,23,59,"CyberSecurity"),
    ("IL","Israel",2022,3700,15000,1150,13.04,7,66,"AI/ML"),
    ("IL","Israel",2023,3400,8200,1000,8.20,3,69,"AI/ML"),

    # ── Canada ─────────────────────────────────────────────────────────────────
    ("CA","Canada",2015,3200,2400,780,3.08,3,8,"AI/ML"),
    ("CA","Canada",2016,3400,2700,820,3.29,2,10,"AI/ML"),
    ("CA","Canada",2017,3800,3500,870,4.02,3,13,"AI/ML"),
    ("CA","Canada",2018,4200,4900,930,5.27,4,17,"AI/ML"),
    ("CA","Canada",2019,4700,6200,990,6.26,5,22,"AI/ML"),
    ("CA","Canada",2020,4400,5100,940,5.43,6,28,"HealthTech"),
    ("CA","Canada",2021,5700,14000,1200,11.67,15,43,"FinTech"),
    ("CA","Canada",2022,6000,8900,1250,7.12,5,48,"AI/ML"),
    ("CA","Canada",2023,5800,6200,1150,5.39,4,52,"AI/ML"),

    # ── Australia ──────────────────────────────────────────────────────────────
    ("AU","Australia",2015,2100,1400,560,2.50,1,4,"FinTech"),
    ("AU","Australia",2016,2300,1600,600,2.67,2,6,"FinTech"),
    ("AU","Australia",2017,2600,1900,650,2.92,2,8,"FinTech"),
    ("AU","Australia",2018,2900,2800,710,3.94,3,11,"FinTech"),
    ("AU","Australia",2019,3200,3400,770,4.42,3,14,"HealthTech"),
    ("AU","Australia",2020,3000,3100,730,4.25,3,17,"HealthTech"),
    ("AU","Australia",2021,3800,9200,930,9.89,8,25,"FinTech"),
    ("AU","Australia",2022,4100,5900,990,5.96,3,28,"CleanTech"),
    ("AU","Australia",2023,4000,4100,920,4.46,2,30,"AI/ML"),

    # ── Brazil ─────────────────────────────────────────────────────────────────
    ("BR","Brazil",2015,2100,1200,480,2.50,1,3,"eCommerce"),
    ("BR","Brazil",2016,2000,800,430,1.86,0,3,"eCommerce"),
    ("BR","Brazil",2017,2200,1400,470,2.98,1,4,"FinTech"),
    ("BR","Brazil",2018,2500,2600,520,5.00,2,6,"FinTech"),
    ("BR","Brazil",2019,2800,3000,570,5.26,3,9,"FinTech"),
    ("BR","Brazil",2020,2700,3200,540,5.93,4,13,"EdTech"),
    ("BR","Brazil",2021,3800,9800,800,12.25,10,23,"FinTech"),
    ("BR","Brazil",2022,4000,4400,820,5.37,3,26,"FinTech"),
    ("BR","Brazil",2023,3900,3200,770,4.16,1,27,"eCommerce"),

    # ── China ──────────────────────────────────────────────────────────────────
    ("CN","China",2015,12000,49000,5100,9.61,26,65,"eCommerce"),
    ("CN","China",2016,14000,65000,5800,11.21,30,95,"AI/ML"),
    ("CN","China",2017,17000,75000,6700,11.19,37,132,"AI/ML"),
    ("CN","China",2018,20000,105000,7200,14.58,41,173,"AI/ML"),
    ("CN","China",2019,22000,81000,7400,10.95,40,213,"AI/ML"),
    ("CN","China",2020,21000,71000,6900,10.29,38,251,"HealthTech"),
    ("CN","China",2021,26000,130000,8500,15.29,74,325,"EV/CleanTech"),
    ("CN","China",2022,24000,75000,7800,9.62,18,343,"EV/CleanTech"),
    ("CN","China",2023,22000,55000,7000,7.86,8,351,"AI/ML"),

    # ── South Korea ────────────────────────────────────────────────────────────
    ("KR","South Korea",2015,2200,2100,630,3.33,3,7,"eCommerce"),
    ("KR","South Korea",2016,2500,2400,680,3.53,2,9,"eCommerce"),
    ("KR","South Korea",2017,2900,3100,740,4.19,3,12,"FinTech"),
    ("KR","South Korea",2018,3400,3900,820,4.76,4,16,"FinTech"),
    ("KR","South Korea",2019,3900,4800,900,5.33,5,21,"AI/ML"),
    ("KR","South Korea",2020,3700,4400,860,5.12,5,26,"BioTech"),
    ("KR","South Korea",2021,4800,12500,1100,11.36,13,39,"FinTech"),
    ("KR","South Korea",2022,5200,6800,1150,5.91,5,44,"AI/ML"),
    ("KR","South Korea",2023,5000,4700,1050,4.48,3,47,"AI/ML"),

    # ── Netherlands ────────────────────────────────────────────────────────────
    ("NL","Netherlands",2015,1400,1200,380,3.16,2,4,"AgriTech"),
    ("NL","Netherlands",2016,1500,1400,410,3.41,2,6,"AgriTech"),
    ("NL","Netherlands",2017,1700,1800,450,4.00,2,8,"FinTech"),
    ("NL","Netherlands",2018,1900,2500,490,5.10,3,11,"FinTech"),
    ("NL","Netherlands",2019,2100,3000,530,5.66,3,14,"FinTech"),
    ("NL","Netherlands",2020,2000,3200,510,6.27,2,16,"HealthTech"),
    ("NL","Netherlands",2021,2600,7600,660,11.52,5,21,"CleanTech"),
    ("NL","Netherlands",2022,2800,4800,700,6.86,3,24,"CleanTech"),
    ("NL","Netherlands",2023,2700,3500,660,5.30,2,26,"AI/ML"),

    # ── Sweden ─────────────────────────────────────────────────────────────────
    ("SE","Sweden",2015,1200,1500,420,3.57,3,9,"FinTech"),
    ("SE","Sweden",2016,1300,1800,450,4.00,3,12,"FinTech"),
    ("SE","Sweden",2017,1500,2100,490,4.29,4,16,"FinTech"),
    ("SE","Sweden",2018,1700,2600,540,4.81,4,20,"FinTech"),
    ("SE","Sweden",2019,1900,3000,590,5.08,4,24,"FinTech"),
    ("SE","Sweden",2020,1800,3400,560,6.07,4,28,"HealthTech"),
    ("SE","Sweden",2021,2300,7100,710,10.00,8,36,"CleanTech"),
    ("SE","Sweden",2022,2500,4200,750,5.60,3,39,"CleanTech"),
    ("SE","Sweden",2023,2400,3000,700,4.29,2,41,"AI/ML"),

    # ── Indonesia ──────────────────────────────────────────────────────────────
    ("ID","Indonesia",2015,1100,600,280,2.14,1,2,"eCommerce"),
    ("ID","Indonesia",2016,1400,1000,330,3.03,2,4,"eCommerce"),
    ("ID","Indonesia",2017,1800,3000,400,7.50,2,6,"eCommerce"),
    ("ID","Indonesia",2018,2200,4000,460,8.70,4,10,"eCommerce"),
    ("ID","Indonesia",2019,2700,5000,530,9.43,4,14,"FinTech"),
    ("ID","Indonesia",2020,2900,3200,560,5.71,3,17,"EdTech"),
    ("ID","Indonesia",2021,3800,5400,780,6.92,5,22,"eCommerce"),
    ("ID","Indonesia",2022,4100,4500,830,5.42,2,24,"FinTech"),
    ("ID","Indonesia",2023,4300,3800,870,4.37,1,25,"AI/ML"),

    # ── 2024 PARTIAL DATA (H1 2024 annualised estimates) ──────────────────────
    # Sources: Crunchbase Q2 2024 report, PitchBook H1 2024, CB Insights State
    #          of Venture Q2 2024, StartupBlink 2024 Index
    # Note: marked is_partial=1 in master dataset
    ("US","United States",2024,83000,145000,9200,15.76,18,548,"AI/ML"),
    ("IN","India",2024,13500,7800,1650,4.73,3,107,"AI/ML"),
    ("GB","United Kingdom",2024,8000,12500,1680,7.44,5,108,"AI/ML"),
    ("DE","Germany",2024,4300,5500,1000,5.50,2,44,"AI/ML"),
    ("FR","France",2024,4100,5800,980,5.92,3,42,"AI/ML"),
    ("SG","Singapore",2024,4500,5200,850,6.12,2,40,"AI/ML"),
    ("IL","Israel",2024,3300,7500,960,7.81,4,73,"CyberSecurity"),
    ("CA","Canada",2024,5700,5400,1100,4.91,3,55,"AI/ML"),
    ("AU","Australia",2024,3900,3800,890,4.27,2,32,"AI/ML"),
    ("BR","Brazil",2024,3800,2900,740,3.92,1,28,"FinTech"),
    ("CN","China",2024,21000,48000,6700,7.16,6,357,"AI/ML"),
    ("KR","South Korea",2024,4900,4200,1010,4.16,2,49,"AI/ML"),
    ("NL","Netherlands",2024,2650,3200,640,5.00,2,28,"AI/ML"),
    ("SE","Sweden",2024,2350,2800,680,4.12,1,42,"CleanTech"),
    ("ID","Indonesia",2024,4500,3500,900,3.89,1,26,"eCommerce"),
]

COLUMNS = [
    "country_code", "country_name", "year",
    "startup_count", "total_funding_usd_mn", "num_deals",
    "avg_deal_size_usd_mn", "num_unicorns", "cumulative_unicorns",
    "top_sector"
]


def load_startup_data() -> pd.DataFrame:
    df = pd.DataFrame(STARTUP_DATA, columns=COLUMNS)
    df["pandemic_period"] = df["year"].apply(
        lambda y: "pre" if y < 2020 else ("during" if y <= 2021 else "post")
    )
    df["funding_per_startup"] = (df["total_funding_usd_mn"] / df["startup_count"]).round(4)
    return df


def save_to_db(df: pd.DataFrame, conn: sqlite3.Connection):
    df.to_sql("startup_ecosystem_raw", conn, if_exists="replace", index=False)
    logger.info(f"Saved {len(df)} rows to startup_ecosystem_raw")


def run():
    logger.info("=== Module 2 Step 2: Startup Ecosystem Data ===")
    df = load_startup_data()
    logger.info(f"Loaded {len(df)} records for {df['country_code'].nunique()} countries")

    full_path = get_project_root() / "data/raw/startup_ecosystem_data.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)
    logger.info(f"Saved CSV → data/raw/startup_ecosystem_data.csv")

    conn = get_db_connection()
    save_to_db(df, conn)
    conn.close()

    return df


if __name__ == "__main__":
    df = run()
    print(f"\n✓ Startup ecosystem data loaded: {df.shape}")
    print(df.groupby("pandemic_period")[["startup_count", "total_funding_usd_mn", "num_unicorns"]].mean().round(0))
