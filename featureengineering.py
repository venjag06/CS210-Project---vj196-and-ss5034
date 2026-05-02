import pandas as pd
import sqlite3
import os

DB_PATH = "housing.db"


# ─────────────────────────────────────────────
# Helper: check if a table already exists in DB
# ─────────────────────────────────────────────
def table_exists(db_path: str, table_name: str) -> bool:
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


# ─────────────────────────────────────────────
# STEP 5: Load housing + cities from DB
# ─────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
housing = pd.read_sql("SELECT * FROM housing", conn)
cities  = pd.read_sql("SELECT * FROM cities",  conn)
conn.close()

print(f"housing: {len(housing)} rows")
print(f"cities:  {len(cities)} rows")


# ─────────────────────────────────────────────
# STEP 6: Merge on region + state (inner join)
#
# Inner join keeps only listings where we have
# matching Census demographic data for that city.
# ─────────────────────────────────────────────
if table_exists(DB_PATH, "merged"):
    print("\n✓ 'merged' table already exists — loading from DB")
    conn = sqlite3.connect(DB_PATH)
    merged = pd.read_sql("SELECT * FROM merged", conn)
    conn.close()
    print(f"  Loaded {len(merged)} rows")

else:
    merged = housing.merge(cities, on=["region", "state"], how="inner")
    merged = merged[merged["median_income"] > 0]
    print(f"\nAfter merge: {len(merged)} rows  "
          f"({len(housing) - len(merged):,} listings dropped — no Census match)")

    # ─────────────────────────────────────────────
    # STEP 7: Feature Engineering
    # ─────────────────────────────────────────────

    # 7a. Price per square foot — normalizes price by size so the
    #     model can compare cheap-but-small vs expensive-but-large
    merged["price_per_sqft"] = merged["price"] / merged["sqfeet"]

    # 7b. Average rent for that region — tells us the "going rate"
    #     for a city, useful for detecting over/under-pricing
    merged["avg_rent_by_region"] = (
        merged.groupby(["region", "state"])["price"].transform("mean")
    )

    # 7c. Unemployment rate — converts raw unemployed count to a
    #     percentage of population so cities are comparable
    merged["unemployment_rate"] = (
        merged["unemployed_count"] / merged["population"]
    ) * 100

    # 7d. Bachelor's degree rate — fraction of population with a
    #     degree; higher education correlates with higher rents
    merged["bachelors_rate"] = (
        merged["bachelors_count"] / merged["population"]
    ) * 100

    # 7e. Round new columns for readability
    for col in ["price_per_sqft", "avg_rent_by_region",
                "unemployment_rate", "bachelors_rate"]:
        merged[col] = merged[col].round(4)

    # Drop rows where engineered features couldn't be computed
    merged.dropna(subset=["price_per_sqft", "unemployment_rate",
                           "bachelors_rate"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    merged.to_sql("merged", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✓ 'merged' table saved — {len(merged)} rows")


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
print("\nColumns:", merged.columns.tolist())
print("\nSample rows:")
print(merged[["region", "state", "price", "sqfeet", "beds", "baths",
              "price_per_sqft", "avg_rent_by_region",
              "median_income", "unemployment_rate", "bachelors_rate",
              "median_rent_census"]].head(10).to_string(index=False))

print("\nBasic stats:")
print(merged[["price", "sqfeet", "price_per_sqft",
              "median_income", "unemployment_rate"]].describe().round(2))

print("\nDone!")

