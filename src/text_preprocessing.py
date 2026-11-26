
import re
import pandas as pd
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# Project directories
DATASET_DIR = Path("dataset")

# Download NLTK stopwords if not already available
nltk.download("stopwords", quiet=True)

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Preserve important negation words
negation_words = {"no", "not", "nor", "never"}

stop_words = stop_words - negation_words

# Tokenizer that preserves words and contractions
tokenizer = RegexpTokenizer(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def clean_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Tokenize text
    tokens = tokenizer.tokenize(text)

    # Remove stopwords while preserving negations
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Join tokens back into cleaned text
    return " ".join(tokens)


def preprocess_datasets():

    print("=" * 55)
    print("          TEXT PREPROCESSING")
    print("=" * 55)

    splits = ["train", "validation", "test"]

    for split in splits:

        file_path = DATASET_DIR / f"{split}.csv"

        print(f"\nProcessing {split} dataset...")

        # Read original dataset
        df = pd.read_csv(file_path)

        # Preserve original text
        df["cleaned_text"] = df["text"].apply(clean_text)

        # Remove records that become empty after cleaning
        df = df[df["cleaned_text"].str.strip() != ""]

        # Save processed dataset separately
        output_path = DATASET_DIR / f"{split}_processed.csv"

        df.to_csv(output_path, index=False, encoding="utf-8")

        print("Original samples:", len(pd.read_csv(file_path)))
        print("Processed samples:", len(df))
        print("Saved:", output_path)

        print("\nSample comparison:")

        for _, row in df.head(3).iterrows():

            print("\nOriginal:")
            print(row["text"])

            print("Cleaned:")
            print(row["cleaned_text"])

            print("Sentiment:")
            print(row["sentiment"])

    print("\n" + "=" * 55)
    print("TEXT PREPROCESSING COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    preprocess_datasets()
