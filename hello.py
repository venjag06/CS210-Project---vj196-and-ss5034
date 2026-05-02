import pandas as pd
import sqlite3
import os
import re
import requests
import kagglehub
from difflib import get_close_matches
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "housing.db"
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

# State abbreviation → FIPS code (needed for Census API)
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56',
}

# Helper: check if a table already exists in the DB
def table_exists(db_path: str, table_name: str) -> bool:
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# Common abbreviation expansions
ABBREV = {
    r'\bft\b':    'fort',
    r'\bst\b':    'saint',
    r'\bmt\b':    'mount',
    r'\bpt\b':    'point',
    r'\bpk\b':    'park',
}

def normalize(text: str) -> str:
    """Lowercase, remove periods, expand abbreviations, strip trailing place-type words."""
    text = text.lower().replace('.', '')
    for pattern, replacement in ABBREV.items():
        text = re.sub(pattern, replacement, text)
    # Strip place-type words so "new york city" and "new york" both become "new york"
    text = re.sub(r'\b(city|metro|area)\b', '', text)
    return text.strip()

# Helper: clean a Census place name for matching
# e.g. "Los Angeles city, California" → "los angeles"
def clean_census_name(name: str) -> str:
    name = name.split(",")[0]  # Remove ", California" suffix
    name = re.sub(r'\b(city|town|village|borough|cdp|municipality|township)\b', '', name, flags=re.IGNORECASE)
    return normalize(name)

# Helper: validate that a region keyword is a genuine whole-word match
# in the census name — catches false matches like "reno" inside "fresno"
def is_valid_match(region_key: str, matched_clean_name: str) -> bool:
    key = normalize(region_key)
    name = normalize(matched_clean_name)
    if ' ' in key:
        return key in name
    return bool(re.search(r'\b' + re.escape(key) + r'\b', name))


# ─────────────────────────────────────────────
# STEP 1 & 2: Load HOUSING table
# Skip if already saved to housing.db
# ─────────────────────────────────────────────
if table_exists(DB_PATH, "housing"):
    print("✓ 'housing' table already exists — loading from DB")
    conn = sqlite3.connect(DB_PATH)
    housing = pd.read_sql("SELECT * FROM housing", conn)
    conn.close()
    print(f"  Loaded {len(housing)} rows")
else:
    print("Downloading Kaggle dataset...")
    dataset_path = kagglehub.dataset_download("austinreese/usa-housing-listings")
    raw_df = pd.read_csv(
        dataset_path + "/housing.csv",
        usecols=["id", "region", "price", "type", "sqfeet", "beds", "baths", "state"]
    )
    print(f"Loaded {len(raw_df)} raw rows")

    housing = raw_df.copy()
    housing.dropna(subset=["price", "sqfeet", "beds", "baths", "state"], inplace=True)
    housing = housing[(housing["price"] >= 100) & (housing["price"] <= 10_000)]
    housing = housing[(housing["sqfeet"] >= 100) & (housing["sqfeet"] <= 10_000)]
    housing.drop_duplicates(subset=["id"], inplace=True)
    housing["state"]  = housing["state"].str.strip().str.upper()
    housing["region"] = housing["region"].str.strip().str.lower()
    housing.reset_index(drop=True, inplace=True)

    conn = sqlite3.connect(DB_PATH)
    housing.to_sql("housing", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✓ 'housing' table saved — {len(housing)} rows")

print(housing.head())


# ─────────────────────────────────────────────
# STEP 3: Get unique city/state pairs
# ─────────────────────────────────────────────
unique_locations = (
    housing[["region", "state"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
print(f"\nUnique city/state combos: {len(unique_locations)}")
states_needed = unique_locations["state"].unique().tolist()
print(f"States needed: {len(states_needed)}")


# ─────────────────────────────────────────────
# STEP 4: Fetch Census ACS data via API
# Skip if 'cities' table already exists in DB
#
# For each state, we call the Census API once
# to get all places, then match to our regions.
#
# ACS variables:
#   B19013_001E → Median household income
#   B01003_001E → Total population
#   B23025_005E → Unemployed civilians
#   B15003_022E → Bachelor's degree holders
#   B25064_001E → Median gross rent
# ─────────────────────────────────────────────

def fetch_census_for_state(state_abbr: str) -> pd.DataFrame:
    """Call Census API and return a DataFrame of all places in that state."""
    fips = STATE_FIPS.get(state_abbr)
    if not fips:
        return pd.DataFrame()

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "NAME,B19013_001E,B01003_001E,B23025_005E,B15003_022E,B25064_001E",
        "for": "place:*",
        "in":  f"state:{fips}",
        "key": CENSUS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        headers = data[0]
        rows    = data[1:]
        df = pd.DataFrame(rows, columns=headers)
        df["state_abbr"] = state_abbr
        return df
    except Exception as e:
        print(f"  Census API error for {state_abbr}: {e}")
        return pd.DataFrame()


def build_cities_df(unique_locations: pd.DataFrame) -> pd.DataFrame:
    states = unique_locations["state"].unique().tolist()
    print(f"\nFetching Census data for {len(states)} states via API...")

    all_census = []
    for state in states:
        print(f"  Fetching {state}...", end=" ")
        df = fetch_census_for_state(state)
        if not df.empty:
            print(f"{len(df)} places found")
            all_census.append(df)
        else:
            print("no data")

    if not all_census:
        print("No Census data retrieved!")
        return pd.DataFrame()

    census_all = pd.concat(all_census, ignore_index=True)

    # Clean Census place names for matching
    census_all["clean_name"] = census_all["NAME"].apply(clean_census_name)

    # Match each Craigslist region to the closest Census place name
    census_rows = []
    for _, row in unique_locations.iterrows():
        region = row["region"]   # e.g. "los angeles"
        state  = row["state"]    # e.g. "CA"

        # Only look at places in the right state
        state_places = census_all[census_all["state_abbr"] == state]
        if state_places.empty:
            continue

        candidates = state_places["clean_name"].tolist()

        # Build a list of keys to try, in priority order.
        # For multi-city combos like "spokane / coeur d'alene" or "kennewick-pasco-richland"
        # we try every individual city name, not just the first one.
        all_parts = []
        for slash_part in [p.strip() for p in region.split("/")]:
            for dash_part in [p.strip() for p in slash_part.split("-")]:
                if dash_part:
                    all_parts.append(dash_part)
            if slash_part:
                all_parts.append(slash_part)
        all_parts.append(region)
        keys_to_try = dict.fromkeys(all_parts)  # preserves order, dedupes

        matched_name = None
        matched_key  = None

        # Layer 1: fuzzy match + word-boundary validation
        for key in keys_to_try:
            hits = get_close_matches(normalize(key), [normalize(c) for c in candidates], n=1, cutoff=0.6)
            if hits:
                norm_to_orig = {normalize(c): c for c in candidates}
                orig = norm_to_orig.get(hits[0])
                if orig and is_valid_match(key, orig):
                    matched_name = orig
                    matched_key  = key
                    break

        # Layer 2: regex substring fallback
        # Catches cases like "nashville" inside "nashville-davidson metropolitan government"
        # Word boundaries prevent false positives (e.g. "reno" won't match "fresno")
        if not matched_name:
            for key in keys_to_try:
                pattern = r'\b' + re.escape(normalize(key)) + r'\b'
                for candidate in candidates:
                    if re.search(pattern, normalize(candidate)):
                        matched_name = candidate
                        matched_key  = key
                        break
                if matched_name:
                    break

        if matched_name:
            matched_row = state_places[state_places["clean_name"] == matched_name].iloc[0]
            census_rows.append({
                "region":             region,
                "state":              state,
                "median_income":      matched_row["B19013_001E"],
                "population":         matched_row["B01003_001E"],
                "unemployed_count":   matched_row["B23025_005E"],
                "bachelors_count":    matched_row["B15003_022E"],
                "median_rent_census": matched_row["B25064_001E"],
                "census_matched_to":  matched_row["NAME"],
            })
        else:
            print(f"  No match for: '{region}', {state}")

    cities = pd.DataFrame(census_rows)

    # Convert numeric columns (Census API returns everything as strings)
    numeric_cols = ["median_income", "population", "unemployed_count",
                    "bachelors_count", "median_rent_census"]
    for col in numeric_cols:
        cities[col] = pd.to_numeric(cities[col], errors="coerce")

    cities.dropna(subset=["median_income"], inplace=True)
    cities.reset_index(drop=True, inplace=True)
    return cities


if table_exists(DB_PATH, "cities"):
    print("\n✓ 'cities' table already exists — loading from DB")
    conn = sqlite3.connect(DB_PATH)
    cities = pd.read_sql("SELECT * FROM cities", conn)
    conn.close()
    print(f"  Loaded {len(cities)} rows")
else:
    cities = build_cities_df(unique_locations)
    conn = sqlite3.connect(DB_PATH)
    cities.to_sql("cities", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✓ 'cities' table saved — {len(cities)} rows")

print(cities.head())
print("\nDone!")
