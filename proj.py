import kagglehub 
from kagglehub import KaggleDatasetAdapter

file_path = "data.csv"  # Specify the path to your dataset file within the Kaggle dataset

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "austinreese/usa-housing-listings",
  file_path)

print("First 5 records:", df.head())

if __name__ == "__main__":
    print("Hello, World!")

