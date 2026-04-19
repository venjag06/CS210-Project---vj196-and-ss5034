import pandas as pd
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set Kaggle API credentials from environment variable (if provided)
# Otherwise, kagglehub will automatically use ~/.kaggle/kaggle.json
if "KAGGLE_API_KEY" in os.environ:
    os.environ["KAGGLE_API_KEY"] = os.environ["KAGGLE_API_KEY"]

# Load the dataset
dataset_path = kagglehub.dataset_download("austinreese/usa-housing-listings")
print(f"Dataset downloaded to: {dataset_path}")

df = pd.read_csv(dataset_path + "/housing.csv")  
print(df.head())
print("successs")
if __name__ == "__main__":
    print("Hello, World!")


