import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("austinreese/usa-housing-listings")

print("Path to dataset files:", path)


import pandas as pd
import os

# List files in the downloaded folder
print(os.listdir(path))

# Load the CSV
df = pd.read_csv(path + "/housing.csv")  # replace with the actual filename shown
print(df.head())