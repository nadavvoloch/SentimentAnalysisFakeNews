import os
import pandas as pd
from transformers import pipeline
from flair.data import Sentence as FlairSentence
from flair.models import TextClassifier

# Disable symlink warning for Hugging Face on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize the Hugging Face transformer pipeline for sentiment analysis
hf_sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# Load the Flair sentiment classifier
flair_classifier = TextClassifier.load('sentiment')


def truncate_text(text, max_length=512):
    # Function to truncate text to fit within max token length for models
    return text[:max_length]


fileName = input("Enter file name (Example: file1.csv):")
try:
    df = pd.read_csv(fileName, encoding='utf-8', encoding_errors='ignore')
except Exception as e:
    print(str(e))

lstTextF = df["text"].tolist()
counterFlairPos = 0
counterFlairNeg = 0
counterHFPos = 0
counterHFNeg = 0
SumFlairConfPos = 0
SumFlairConfNeg = 0
SumHFConfPos = 0
SumHFConfNeg = 0
counter = 0

for index, item in enumerate(lstTextF):
    try:
        # Truncate text for Hugging Face model
        truncated_text = truncate_text(item)

        # Hugging Face Transformers Sentiment Analysis
        hf_result = hf_sentiment_analyzer(truncated_text)[0]
        hf_label = hf_result['label']  # Positive or Negative
        hf_score = hf_result['score']  # Confidence score indicating sentiment intensity

        if hf_label == 'NEGATIVE':
            counterHFNeg += 1
            SumHFConfNeg += hf_score
        if hf_label == 'POSITIVE':
            counterHFPos += 1
            SumHFConfPos += hf_score

        # Truncate text for Flair if necessary
        flair_sentence = FlairSentence(truncated_text)
        flair_classifier.predict(flair_sentence)

        flair_label = flair_sentence.labels[0].value  # "POSITIVE" or "NEGATIVE"
        flair_confidence = flair_sentence.labels[0].score  # Confidence score as sentiment score

        if flair_label == 'NEGATIVE':
            counterFlairNeg += 1
            SumFlairConfNeg += flair_confidence
        if flair_label == 'POSITIVE':
            counterFlairPos += 1
            SumFlairConfPos += flair_confidence

        counter += 1
        print("Processing row no.", counter)
    except Exception as ee:
        print(f"Error processing row {index}: {str(ee)}")

print("Flair total Negatives:", counterFlairNeg)
print("Flair total Positives:", counterFlairPos)
print("Hugging Face Transformers total Negatives:", counterHFNeg)
print("Hugging Face Transformers total Positives:", counterHFPos)
print("Total records processed:", counter)
print("Flair average Negative confidence:", SumFlairConfNeg / counterFlairNeg if counterFlairNeg > 0 else 0)
print("Flair average Positive confidence:", SumFlairConfPos / counterFlairPos if counterFlairPos > 0 else 0)
print("Hugging Face Transformers average Negative confidence:", SumHFConfNeg / counterHFNeg if counterHFNeg > 0 else 0)
print("Hugging Face Transformers average Positive confidence:", SumHFConfPos / counterHFPos if counterHFPos > 0 else 0)
