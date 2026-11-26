
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("       LOGISTIC REGRESSION MODEL TRAINING")
print("=" * 65)

# Load TF-IDF feature matrices
print("\nLoading feature matrices...")

X_train = load_npz(MODEL_DIR / "X_train.npz")
X_validation = load_npz(MODEL_DIR / "X_validation.npz")

# Load corresponding labels
y_train = np.load(
    MODEL_DIR / "y_train.npy",
    allow_pickle=True
)

y_validation = np.load(
    MODEL_DIR / "y_validation.npy",
    allow_pickle=True
)

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_validation.shape)

# Compare two model configurations
configurations = {
    "standard": LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver="lbfgs"
    ),

    "balanced": LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced"
    )
}

results = []

best_model = None
best_f1 = -1
best_name = None

for name, model in configurations.items():

    print("\n" + "=" * 65)
    print(f"Training configuration: {name.upper()}")
    print("=" * 65)

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_validation,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    print(f"Validation Accuracy : {accuracy:.4f}")
    print(f"Macro Precision     : {precision:.4f}")
    print(f"Macro Recall        : {recall:.4f}")
    print(f"Macro F1-score      : {macro_f1:.4f}")

    results.append({
        "configuration": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_name = name

# Save the selected model
model_path = MODEL_DIR / "sentiment_model.joblib"

joblib.dump(best_model, model_path)

# Save comparison results
with open(
    MODEL_DIR / "training_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("LOGISTIC REGRESSION TRAINING RESULTS\n")
    file.write("=" * 50 + "\n\n")

    for result in results:
        file.write(
            f"{result['configuration'].upper()}\n"
        )
        file.write(
            f"Accuracy: {result['accuracy']:.4f}\n"
        )
        file.write(
            f"Macro F1: {result['macro_f1']:.4f}\n\n"
        )

    file.write(f"Selected model: {best_name}\n")
    file.write(f"Best validation macro F1: {best_f1:.4f}\n")

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED")
print("=" * 65)

print("Selected configuration:", best_name)
print(f"Best validation macro F1: {best_f1:.4f}")
print("Saved model:", model_path)

print("\nFinal validation classification report:")

validation_predictions = best_model.predict(X_validation)

print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=["negative", "neutral", "positive"],
        zero_division=0
    )
)

