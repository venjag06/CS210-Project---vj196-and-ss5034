import pandas as pd
import os
import kagglehub
from sqlalchemy import create_engine
from census_lookup import CensusLookup

# Set Kaggle API credentials from environment variable (if provided)
# Otherwise, kagglehub will automatically use ~/.kaggle/kaggle.json
if "KAGGLE_API_KEY" in os.environ:
    os.environ["KAGGLE_API_KEY"] = os.environ["KAGGLE_API_KEY"]

# Load the dataset
dataset_path = kagglehub.dataset_download("austinreese/usa-housing-listings")

print(f"Dataset downloaded to: {dataset_path}")

df = pd.read_csv(dataset_path + "/housing.csv")  
print(df.head())
print(f"Loaded {len(df)} rows")

lookup = CensusLookup(
    variables=["B19013_001E", "B17001_001E", "B23025_005E", "B15003_022E", "B25064_001E"],  # Median household income, Total population, # of peopel with bachelors, median rent
)

#cities [# Median household income, Total population, # of peopel with bachelors, median rent]
#housing ['id', 'region', 'price', 'type', 'sqfeet', 'beds', 'baths', 'state']







"""
# SQLite3 Database (no server needed!)
# Database file will be created in your project directory
db_path = os.getenv("DB_PATH", "housing.db")
try:
    # Create SQLite3 connection
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)
    
    # Load dataframe into SQLite3 in chunks (for large files)
    table_name = "housing_listings"
    print(f"Loading rows into SQLite3...")
    df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=10000)
    print(f"✓ Successfully loaded {len(df)} rows into SQLite3 database '{db_path}'")
    
    # IMPORTANT: Close the engine to release database lock
    engine.dispose()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    print("Housing data processing complete!")
    
    # Query and print data from SQLite3
    print("\n" + "="*80)
    print("HOUSING DATABASE CONTENTS")
    print("="*80 + "\n")
    
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT COUNT(*) FROM housing_listings")
        total_rows = cursor.fetchone()[0]
        print(f"Total rows in database: {total_rows}\n")
        
        # Get column names
        cursor.execute("PRAGMA table_info(housing_listings)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Columns: {columns}\n")
        
        # Print first 10 rows
        print("First 10 rows:")
        cursor.execute("SELECT * FROM housing_listings LIMIT 10")
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"{i}: {row}")
        
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
        """