
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import load_npz

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay
)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"

print("=" * 65)
print("             FINAL MODEL EVALUATION")
print("=" * 65)

# Load the saved model and untouched test data
print("\nLoading model and test data...")

model = joblib.load(MODEL_PATH)
X_test = load_npz(MODEL_DIR / "X_test.npz")
y_test = np.load(MODEL_DIR / "y_test.npy", allow_pickle=True)

print("Test matrix:", X_test.shape)
print("Test labels:", len(y_test))

# Generate predictions
print("\nGenerating predictions...")
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

precision, recall, macro_f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 65)
print("                 TEST PERFORMANCE")
print("=" * 65)

print(f"Test Accuracy  : {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1-score : {macro_f1:.4f}")

# Classification report
class_names = ["negative", "neutral", "positive"]

report = classification_report(
    y_test,
    y_pred,
    labels=class_names,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# Save evaluation report
with open(MODEL_DIR / "evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write("SENTIMENT ANALYSIS - FINAL TEST EVALUATION\n")
    file.write("=" * 50 + "\n\n")
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
    file.write(report)

# Generate and save confusion matrix
print("Generating confusion matrix...")

display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=class_names,
    display_labels=class_names,
    cmap="Blues",
    values_format="d"
)

display.ax_.set_title("Sentiment Analysis - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nEvaluation report saved:")
print("models/evaluation_report.txt")

print("Confusion matrix saved:")
print("models/confusion_matrix.png")

print("\nFINAL EVALUATION COMPLETED")
