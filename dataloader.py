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
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY") #cant be putting my api key on a public repo lol

# check if a table already exists in the housing.db sqllite3 database, 
# if it does then load it instead of re-downloading and processing the data from scratch
# this saves a lot of computation 
def table_exists(db_path: str, table_name: str) -> bool:
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    result = cursor.fetchone()
    conn.close()
    return result is not None



#trying to normalize and clean names before matching in census data
def normalize(text: str) -> str:
    text = text.split(",")[0]  # Remove ", California" suffix
    text = re.sub(r'\b(city|town|village|borough|cdp|municipality|township)\b', '', text, flags=re.IGNORECASE)
    # common abbreviation expansions (e.g. "ft" → "fort", "st" → "saint") to improve matching accuracy 
    ABBREV = {
        r'\bft\b':    'fort',
        r'\bst\b':    'saint',
        r'\bmt\b':    'mount',
        r'\bpt\b':    'point',
        r'\bpk\b':    'park',
    }
    # Lowercase, remove periods, expand abbreviations, strip trailing place-type words
    text = text.lower().replace('.', '')
    for pattern, replacement in ABBREV.items():
        text = re.sub(pattern, replacement, text)
    # Strip place-type words so "new york city" and "new york" both become "new york"
    text = re.sub(r'\b(city|metro|area)\b', '', text)
    return text.strip()


# Helper: validate that a region keyword is a genuine whole-word match
# in the census name — catches false matches like "reno" inside "fresno"
def is_valid_match(region_key: str, matched_clean_name: str) -> bool:
    key = normalize(region_key)
    name = normalize(matched_clean_name)
    if ' ' in key:
        return key in name
    return bool(re.search(r'\b' + re.escape(key) + r'\b', name))


#loading housing table 

if table_exists(DB_PATH, "housing"):
# if the table already exists in the database, load it instead of re-downloading and processing the data from scratch
    print("'housing' table already exists — loading from DB")
    conn = sqlite3.connect(DB_PATH)
    housing = pd.read_sql("SELECT * FROM housing", conn)
    conn.close()
    print(f"  Loaded {len(housing)} rows")
else:
    print("Downloading Kaggle dataset...")
    dataset_path = kagglehub.dataset_download("austinreese/usa-housing-listings") #this is the dataset we are using
    raw_df = pd.read_csv(
        dataset_path + "/housing.csv",
        usecols=["id", "region", "price", "type", "sqfeet", "beds", "baths", "state"]
    )
    #loading it into a pandas dataframe with just the columns we need. so we don't have to drop them later. 
    print(f"Loaded {len(raw_df)} raw rows")

    #making a new dataframe called housing that we will clean and process, so we can always refer back to the raw_df if we need to
    housing = raw_df.copy()
    #dropping any rows with missing values in critical columns
    housing.dropna(subset=["price", "sqfeet", "beds", "baths", "state"], inplace=True)
    #trying to avoid extreme outliers by only considering housings between $100 and $10,000 per month
    housing = housing[(housing["price"] >= 100) & (housing["price"] <= 10_000)]
    #trying to avoid extreme outliers by only considering housings between 100 and 10,000 square feet
    housing = housing[(housing["sqfeet"] >= 100) & (housing["sqfeet"] <= 10_000)]
    #dropping duplicate listings based on the unique 'id' column provided in the dataset
    housing.drop_duplicates(subset=["id"], inplace=True)
    #cleaning the 'state' and 'region' column by stripping whitespace and converting to uppercase for consistency (ex. "ca" becomes "CA")
    housing["state"]  = housing["state"].str.strip().str.upper()
    housing["region"] = housing["region"].str.strip().str.lower()
    housing.reset_index(drop=True, inplace=True)
    
    # connect to database and save the cleaned housing dataframe as a new table called 'housing'
    conn = sqlite3.connect(DB_PATH)
    housing.to_sql("housing", conn, if_exists="replace", index=False)
    conn.close()
    print(f"'housing' table saved — {len(housing)} rows")

print(housing.head())


#getting unique city/state pairs
unique_locations = (
    housing[["region", "state"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
print(f"\nUnique city/state combos: {len(unique_locations)}")
states_needed = unique_locations["state"].unique().tolist()
print(f"States needed: {len(states_needed)}")
# once we get the unique city/state pairs, we can use that to query the census api for the relevant cities and states, 
# instead of querying for every single listing in the housing data which would be very inefficient 
# and could get us rate limited by the census api.


#Fetch Census ACS data via API
# Skip if 'cities' table already exists in DB
# For each state, we call the Census API once
# to get all places, then match to our regions.
#
# ACS variables from their documentation:
#   B19013_001E → Median household income
#   B01003_001E → Total population
#   B23025_005E → Unemployed civilians
#   B15003_022E → Bachelor's degree holders
#   B25064_001E → Median gross rent
# ─────────────────────────────────────────────

def fetch_census_for_state(state_abbr: str) -> pd.DataFrame:
    #Call Census API and return a DataFrame of all places in that state.

    # State abbreviation to FIPS code, needed for Census API querying
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
    fips = STATE_FIPS.get(state_abbr)
    if not fips:
        return pd.DataFrame()

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "NAME,B19013_001E,B01003_001E,B23025_005E,B15003_022E,B25064_001E",
        "for": "place:*",
        "in":  f"state:{fips}",
        "key": CENSUS_API_KEY,
    } #setting up request headers to query api
    try:
        #making the request and parsing the response into DataFrame and returning that df
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
    #using the unique city/state pairs to get a list of unique states that we need to query the census api for
    states = unique_locations["state"].unique().tolist() 
    print(f"\nFetching Census data for {len(states)} states via API...")

    all_census = []
    """for each state we call the census api to get all the cities in that state, and then we will match those cities to the regions 
    in our housing data using fuzzy matching and regex. this way we only call the census api once per state, 
    instead of once per listing which would be very inefficient and could get us rate limited by the census api."""

    for state in states:
        print(f"  Fetching {state}...", end=" ")
        df = fetch_census_for_state(state) #fetch census data for that state
        if not df.empty:
            print(f"{len(df)} places found")
            all_census.append(df)
        else:
            print("no data")

    if not all_census:
        print("No Census data retrieved!")
        return pd.DataFrame()

    census_all = pd.concat(all_census, ignore_index=True) 
    #census_all is a dataframe of all the census data we retrieved for all the states we needed, 
    # with columns for median income, population, unemployed count, bachelors count, median rent, and state abbreviation

    # Clean Census place names for matching using the normalize function from above
    census_all["clean_name"] = census_all["NAME"].apply(normalize)

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

        # Layer 1: fuzzy match + word-boundary validation
        for key in keys_to_try:
            #using the difflib get_close_matches function to find the closest match for the region key
            # in the list of census place names, with a cutoff of 0.6 for similarity
            hits = get_close_matches(normalize(key), [normalize(c) for c in candidates], n=1, cutoff=0.6)
            if hits:
                norm_to_orig = {normalize(c): c for c in candidates}
                orig = norm_to_orig.get(hits[0])
                if orig and is_valid_match(key, orig):
                    matched_name = orig
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

    # Drop rows where we couldn't get valid numeric data for median income, since that's a critical feature
    cities.dropna(subset=["median_income"], inplace=True)
    cities.reset_index(drop=True, inplace=True)
    return cities


#if cities table already exists load from db, if not then build the dataframe and load it into our sql database as a new table called 'cities' for future use so we don't have to call the census api again and do all the processing again next time we run the code. this saves a lot of time and computation.
if table_exists(DB_PATH, "cities"):
    print("\n'cities' table already exists — loading from DB")
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
print("\nloaded")
