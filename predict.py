import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import re
from difflib import get_close_matches

DB_PATH = "housing.db"
#loading saved models
try:
    lr_model    = joblib.load("models/lr_model.joblib")
    lr_scaler   = joblib.load("models/lr_scaler.joblib")
    knn_model   = joblib.load("models/knn_model.joblib")
    knn_scaler  = joblib.load("models/knn_scaler.joblib")
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)
except FileNotFoundError:
    print("ERROR: Model files not found.")
    print("Please run linear_regression.py and knn_regression.py first.")
    exit(1)

# Load cities and region averages from DB to print out 
conn = sqlite3.connect(DB_PATH)
cities  = pd.read_sql("SELECT * FROM cities", conn)
merged  = pd.read_sql("SELECT region, state, AVG(price) as avg_rent FROM merged GROUP BY region, state", conn)
conn.close()

# Valid property types
type_cols   = [c for c in feature_cols if c.startswith("type_")]
valid_types = [c.replace("type_", "") for c in type_cols]

ALL_TYPES = ["apartment", "house", "condo", "townhouse", "duplex",
             "flat", "in-law", "loft", "cottage/cabin", "manufactured"]


# word normalizer
def normalize(text: str) -> str:
    text = text.lower().replace(".", "").strip()
    text = re.sub(r"\b(city|metro|area)\b", "", text)
    return text.strip()

# to match a city that doesn't have an exact match
def find_city(user_city: str, user_state: str) -> pd.Series | None:
    """Fuzzy-match user input to a region in our cities table."""
    state_cities = cities[cities["state"] == user_state.upper()]
    if state_cities.empty:
        return None

    candidates = state_cities["region"].tolist()
    key = normalize(user_city)

    hits = get_close_matches(key, [normalize(c) for c in candidates], n=1, cutoff=0.5)
    if hits:
        norm_to_orig = {normalize(c): c for c in candidates}
        matched = norm_to_orig.get(hits[0])
        if matched:
            return state_cities[state_cities["region"] == matched].iloc[0]

    for c in candidates:
        if key in normalize(c) or normalize(c) in key:
            return state_cities[state_cities["region"] == c].iloc[0]

    return None

# fetching average rent 
def get_avg_rent(region: str, state: str) -> float | None:
    row = merged[(merged["region"] == region) & (merged["state"] == state)]
    return round(row["avg_rent"].values[0], 2) if not row.empty else None

def prompt(label: str, cast=str, valid=None, optional=False) -> any:
    """Ask the user for input, re-prompt on bad input."""
    while True:
        raw = input(f"  {label}: ").strip()
        if not raw and optional:
            return None
        try:
            val = cast(raw)
            if valid and val not in valid:
                print(f"    Please enter one of: {', '.join(str(v) for v in valid)}")
                continue
            return val
        except (ValueError, TypeError):
            print(f"    Invalid input, please try again.")

def build_feature_row(sqfeet, beds, baths, prop_type,
                      median_income, unemployment_rate,
                      bachelors_rate, median_rent_census) -> np.ndarray:
    """Build a single feature vector matching the training columns."""
    row = {col: 0 for col in feature_cols}
    row["sqfeet"]             = sqfeet
    row["beds"]               = beds
    row["baths"]              = baths
    row["median_income"]      = median_income
    row["unemployment_rate"]  = unemployment_rate
    row["bachelors_rate"]     = bachelors_rate
    row["median_rent_census"] = median_rent_census

    # Set the type dummy if it exists as a column =
    type_col = f"type_{prop_type}"
    if type_col in row:
        row[type_col] = 1

    return np.array([row[c] for c in feature_cols]).reshape(1, -1)


print("\n" + "═" * 50)
print("       Rental Price Predictor")
print("═" * 50)


# location
print("── Location ─────────────────────────────────")
while True:
    city  = input("  City (e.g. austin, los angeles): ").strip()
    state = input("  State abbreviation (e.g. TX, CA):  ").strip().upper()
    #matching city 
    city_row = find_city(city, state)
    if city_row is not None:
        print(f"\n  ✓ Matched to: {city_row['region'].title()}, {state}")
        break
    print(f"\n  ✗ Could not find '{city}' in {state}.")
    print("    Make sure the city is in the Craigslist dataset and try again.\n")

region = city_row["region"]
avg_rent = get_avg_rent(region, state)

# property information
print("\n── Property Details ─────────────────────────")
sqfeet    = prompt("Square footage", cast=float)
beds      = prompt("Bedrooms",       cast=int)
baths     = prompt("Bathrooms",      cast=float)

print(f"  Property type options: {', '.join(ALL_TYPES)}")
prop_type = prompt("Property type", cast=str)
prop_type = prop_type.lower().strip()

# listing price (optional)  for compariosnn purpo
print("\n── Comparison (optional) ────────────────────")
listed_price_raw = input("  Listed price (press Enter to skip): ").strip()
listed_price = float(listed_price_raw) if listed_price_raw else None


#building feature row and actually predicting
X_input = build_feature_row(
    sqfeet         = sqfeet,
    beds           = beds,
    baths          = baths,
    prop_type      = prop_type,
    median_income      = city_row["median_income"],
    unemployment_rate  = city_row["unemployment_rate"] if "unemployment_rate" in city_row else 0,
    bachelors_rate     = city_row["bachelors_rate"]    if "bachelors_rate"    in city_row else 0,
    median_rent_census = city_row["median_rent_census"],
)

lr_pred  = np.exp(lr_model.predict(lr_scaler.transform(X_input))[0])
knn_pred = np.exp(knn_model.predict(knn_scaler.transform(X_input))[0])


# displaying result
print("\n" + "═" * 50)
print("  Results")
print("═" * 50)
print(f"  City:          {region.title()}, {state}")
print(f"  Property:      {beds}bd / {baths}ba  |  {sqfeet:.0f} sqft  |  {prop_type}")
print()
print(f"  Linear Regression prediction:  ${lr_pred:>8,.2f}")
print(f"  KNN prediction:                ${knn_pred:>8,.2f}")
if avg_rent:
    print(f"  Avg listed rent in this city:  ${avg_rent:>8,.2f}")
print()

if listed_price:
    lr_diff   = listed_price - lr_pred
    knn_diff  = listed_price - knn_pred
    lr_pct    = lr_diff  / lr_pred  * 100
    knn_pct   = knn_diff / knn_pred * 100

    print(f"  Your listed price:  ${listed_price:,.2f}")
    print()

    def diff_line(label, diff, pct):
        direction = "OVER" if diff > 0 else "UNDER"
        sign      = "+" if diff > 0 else ""
        return f"  vs {label:<25} {sign}${abs(diff):,.2f}  ({sign}{pct:.1f}%)  [{direction}]"

    print(diff_line("LR prediction:",  lr_diff,  lr_pct))
    print(diff_line("KNN prediction:", knn_diff, knn_pct))
    print()

    both_over  = lr_diff > 0 and knn_diff > 0
    both_under = lr_diff < 0 and knn_diff < 0
    if both_over:
        print("  Verdict: Both models suggest this listing is OVERPRICED.")
    elif both_under:
        print("  Verdict: Both models suggest this listing is UNDERPRICED (a potential deal).")
    else:
        print("  Verdict: Models disagree — the listing is near the predicted fair price.")

print("═" * 50 + "\n")