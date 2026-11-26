
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# Project directories
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

# Dataset splits
splits = ["train", "validation", "test"]

print("=" * 60)
print("          TF-IDF FEATURE EXTRACTION")
print("=" * 60)

# Load processed datasets
datasets = {}

for split in splits:

    file_path = DATASET_DIR / f"{split}_processed.csv"

    df = pd.read_csv(file_path)

    datasets[split] = df

    print(f"{split.capitalize()} samples: {len(df)}")


# Initialize TF-IDF
vectorizer = TfidfVectorizer(
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    sublinear_tf=True
)


print("\nFitting TF-IDF on training data...")

# Fit ONLY on training text
X_train = vectorizer.fit_transform(
    datasets["train"]["cleaned_text"]
)

# Transform validation and test data
X_validation = vectorizer.transform(
    datasets["validation"]["cleaned_text"]
)

X_test = vectorizer.transform(
    datasets["test"]["cleaned_text"]
)


# Save the vectorizer
joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)


# Save feature matrices
save_npz(MODEL_DIR / "X_train.npz", X_train)
save_npz(MODEL_DIR / "X_validation.npz", X_validation)
save_npz(MODEL_DIR / "X_test.npz", X_test)


# Save corresponding labels
for split in splits:

    labels = datasets[split]["sentiment"].to_numpy()

    np.save(
        MODEL_DIR / f"y_{split}.npy",
        labels
    )


print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print("\nFeature matrix dimensions:")

print("Training   :", X_train.shape)
print("Validation :", X_validation.shape)
print("Testing    :", X_test.shape)

print("\nTotal unique features:", len(vectorizer.get_feature_names_out()))

print("\nVectorizer saved successfully.")
print("Feature matrices saved successfully.")
print("Labels saved successfully.")
