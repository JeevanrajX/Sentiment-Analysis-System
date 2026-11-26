
import re
import joblib

from pathlib import Path
from flask import Flask, render_template, request

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

# --------------------------------------------------
# LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# --------------------------------------------------

model_path = MODEL_DIR / "sentiment_model.joblib"
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("Sentiment model loaded successfully!")
print("TF-IDF vectorizer loaded successfully!")


# --------------------------------------------------
# TEXT PREPROCESSING
# Must match training preprocessing
# --------------------------------------------------

def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Tokenize text
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)

    # English stopwords
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    # Preserve negation words
    stop_words -= {"no", "not", "nor", "never"}

    cleaned_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    probabilities = None
    original_text = ""

    if request.method == "POST":

        original_text = request.form.get("text", "").strip()

        if original_text:

            cleaned_text = preprocess_text(original_text)

            if cleaned_text:

                features = vectorizer.transform([cleaned_text])

                prediction = model.predict(features)[0]

                result = str(prediction).lower()

                probability_values = model.predict_proba(features)[0]

                probabilities = {
                    str(label).lower(): round(float(probability) * 100, 2)
                    for label, probability in zip(
                        model.classes_,
                        probability_values
                    )
                }

            else:
                result = "unable_to_analyze"

        else:
            result = "empty_input"

    return render_template(
        "index.html",
        result=result,
        probabilities=probabilities,
        original_text=original_text
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
