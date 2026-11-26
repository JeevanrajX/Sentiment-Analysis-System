\# Sentiment Analysis System



A Machine Learning-based web application that analyzes text and classifies its sentiment into three categories: \*\*Positive, Negative, and Neutral\*\*.



The application uses \*\*TF-IDF for text feature extraction\*\* and \*\*Logistic Regression for sentiment classification\*\*, with a Flask-powered web interface.



\## Application Preview



!\[Sentiment Analysis Web Application](screenshots/sentiment-analysis.png)



\## Project Overview



The Sentiment Analysis System is designed to identify the emotional polarity of textual input using Natural Language Processing (NLP) and traditional Machine Learning techniques.



Users can enter a sentence, review, or short text and receive a predicted sentiment along with the model's prediction probabilities.



\### Key Features



\- Three-class sentiment classification: Positive, Negative, and Neutral.

\- Text preprocessing and cleaning using NLTK.

\- TF-IDF feature extraction with unigrams and bigrams.

\- Logistic Regression classification.

\- Comparison of standard and class-balanced model configurations.

\- Interactive Flask web application.

\- Prediction probability visualisation.

\- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.



\## Technologies Used



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| Pandas \& NumPy | Data handling |

| NLTK | Text preprocessing |

| Scikit-learn | Machine Learning |

| TF-IDF | Text feature extraction |

| Logistic Regression | Sentiment classification |

| Flask | Web application backend |

| HTML \& CSS | User interface |

| Matplotlib \& Seaborn | Evaluation visualisation |

| Joblib | Model persistence |



\## Dataset



This project uses the \*\*TweetEval Sentiment Dataset\*\* from Hugging Face.



Dataset: \[cardiffnlp/tweet\_eval](https://huggingface.co/datasets/cardiffnlp/tweet\_eval)



The sentiment configuration contains three classes:



| Label | Sentiment |

|---|---|

| 0 | Negative |

| 1 | Neutral |

| 2 | Positive |



\### Dataset Distribution



| Split | Original Records |

|---|---:|

| Training | 45,615 |

| Validation | 2,000 |

| Testing | 12,284 |

| \*\*Total\*\* | \*\*59,899\*\* |



Two test records became empty after preprocessing and were excluded from feature extraction. Therefore, the final test evaluation contains 12,282 records.



\## Project Workflow



1\. \*\*Dataset Collection:\*\* Download the TweetEval sentiment dataset.

2\. \*\*Text Preprocessing:\*\* Convert text to lowercase, remove URLs, usernames, HTML tags and stopwords while preserving negation words.

3\. \*\*Feature Extraction:\*\* Convert text into numerical features using TF-IDF with unigrams and bigrams.

4\. \*\*Model Training:\*\* Train Logistic Regression models using standard and class-balanced configurations.

5\. \*\*Model Selection:\*\* Select the configuration with the highest validation macro F1-score.

6\. \*\*Model Evaluation:\*\* Evaluate the selected model using the unseen test dataset.

7\. \*\*Web Application:\*\* Integrate the trained model with Flask for interactive sentiment predictions.



\## Machine Learning Model



\*\*TF-IDF (Term Frequency–Inverse Document Frequency)\*\* converts textual data into numerical feature vectors.



\*\*Logistic Regression\*\* uses these features to classify text into one of three sentiment categories.



Two configurations were compared:



\- Standard Logistic Regression

\- Class-balanced Logistic Regression



The class-balanced configuration was selected because it achieved the higher validation macro F1-score.



\### Model Selection Results



| Configuration | Validation Accuracy | Validation Macro F1 |

|---|---:|---:|

| Standard | 67.55% | 63.50% |

| Balanced | 67.00% | 65.06% |



\## Final Model Evaluation



The selected model was evaluated using the held-out test dataset.



| Metric | Test Result |

|---|---:|

| Accuracy | 58.52% |

| Macro Precision | 57.60% |

| Macro Recall | 59.53% |

| Macro F1-score | 58.24% |



\### Classification Report



| Sentiment | Precision | Recall | F1-score |

|---|---:|---:|---:|

| Negative | 56% | 65% | 60% |

| Neutral | 64% | 54% | 58% |

| Positive | 53% | 60% | 56% |



\### Confusion Matrix



!\[Confusion Matrix](models/confusion\_matrix.png)



The model evaluation shows moderate performance across all three sentiment categories. The results also highlight opportunities for improving neutral and positive sentiment classification.



\## Project Structure



```text

Sentiment-Analysis-System/

│

├── dataset/

│

├── models/

│   ├── sentiment\_model.joblib

│   ├── tfidf\_vectorizer.joblib

│   ├── confusion\_matrix.png

│   └── evaluation\_report.txt

│

├── src/

│   ├── download\_dataset.py

│   ├── text\_preprocessing.py

│   ├── verify\_preprocessing.py

│   ├── tfidf\_features.py

│   ├── train\_model.py

│   └── evaluate\_model.py

│

├── templates/

│   └── index.html

│

├── static/

│

├── screenshots/

│   └── sentiment-analysis.png

│

├── app.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation and Execution



\### 1. Clone the Repository



```bash

git clone https://github.com/YOUR\_USERNAME/Sentiment-Analysis-System.git

cd Sentiment-Analysis-System

```



\### 2. Create a Virtual Environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\activate

```



\### 3. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 4. Download NLTK Stopwords



```bash

python -c "import nltk; nltk.download('stopwords')"

```



\### 5. Run the Application



```bash

python app.py

```



Open the application in your browser:



http://127.0.0.1:5000



\*\*Note:\*\* The trained model and TF-IDF vectorizer files must be present in the `models/` directory. The raw dataset and intermediate training matrices are not included in this repository. To reproduce training, download the dataset and run the preprocessing, feature extraction, training, and evaluation scripts in order.



\## Future Enhancements



\- Improve classification accuracy through hyperparameter tuning.

\- Experiment with advanced text representation techniques.

\- Explore additional Machine Learning and Deep Learning models.

\- Improve handling of sarcasm and context-dependent expressions.

\- Enhance the user interface and add batch text analysis.



\## Author
