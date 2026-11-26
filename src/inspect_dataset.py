
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("dataset")

files = {
    "Training": "train.csv",
    "Validation": "validation.csv",
    "Testing": "test.csv"
}

print("=" * 60)
print("           SENTIMENT DATASET INSPECTION")
print("=" * 60)

for name, filename in files.items():

    path = DATASET_DIR / filename

    df = pd.read_csv(path)

    print(f"\n{name} Dataset")
    print("-" * 40)

    print("Total records:", len(df))
    print("Missing values:")
    print(df.isnull().sum())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nSample records:")
    print(df[["text", "sentiment"]].head(3).to_string(index=False))

    print("\n" + "-" * 60)

print("\nDataset inspection completed.")
