
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Project dataset directory
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)

# Sentiment label mapping
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def download_sentiment_dataset():

    print("=" * 55)
    print("       SENTIMENT ANALYSIS DATASET DOWNLOAD")
    print("=" * 55)

    print("\nDownloading TweetEval sentiment dataset...")

    # Download official dataset with predefined splits
    dataset = load_dataset(
        "cardiffnlp/tweet_eval",
        "sentiment"
    )

    split_names = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv"
    }

    for split, filename in split_names.items():

        print(f"\nProcessing {split} dataset...")

        # Convert dataset into pandas DataFrame
        df = dataset[split].to_pandas()

        # Convert numerical labels into sentiment names
        df["sentiment"] = df["label"].map(LABEL_MAP)

        # Keep required columns
        df = df[["text", "label", "sentiment"]]

        # Remove missing and empty text records
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip() != ""]

        # Save CSV
        output_path = DATASET_DIR / filename
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"Saved: {output_path}")
        print(f"Total samples: {len(df)}")

        print("\nSentiment distribution:")
        print(df["sentiment"].value_counts())

    print("\n" + "=" * 55)
    print("DATASET DOWNLOAD COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    download_sentiment_dataset()
