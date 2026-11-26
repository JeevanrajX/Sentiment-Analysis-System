
import pandas as pd

df = pd.read_csv("dataset/train_processed.csv")

print("=" * 55)
print("       PREPROCESSING VERIFICATION")
print("=" * 55)

print("\nTotal processed records:", len(df))

print("\nMissing cleaned text:")
print(df["cleaned_text"].isnull().sum())

print("\nSample cleaned sentences:")

for i in range(5):

    print(f"\nSample {i + 1}")
    print("Original :", df.iloc[i]["text"])
    print("Cleaned  :", df.iloc[i]["cleaned_text"])
    print("Sentiment:", df.iloc[i]["sentiment"])

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nVerification completed.")
