import pandas as pd
import os
import kagglehub
from sqlalchemy import create_engine

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

# SQLite3 Database (no server needed!)
# Database file will be created in your project directory
db_path = os.getenv("DB_PATH", "housing.db")

try:
    # Create SQLite3 connection
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)
    
    # Test connection
    with engine.connect() as conn:
        pass  # Just test the connection
    
    # Load dataframe into SQLite3 in chunks (for large files)
    table_name = "housing_listings"
    print(f"Loading {len(df)} rows into SQLite3...")
    df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=10000)
    print(f"✓ Successfully loaded {len(df)} rows into SQLite3 database '{db_path}'")
    print(f"✓ Table created: '{table_name}'")
    print(f"✓ Columns: {list(df.columns)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    print("Housing data processing complete!")