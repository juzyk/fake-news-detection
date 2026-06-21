import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first time)
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_dataset(df):
    """
    Remove missing values, empty texts and duplicates.
    """

    # Remove NaN values
    df = df.dropna(subset=["text", "label"])

    # Convert to string
    df["text"] = df["text"].astype(str)

    # Remove empty rows
    df = df[df["text"].str.strip() != ""]

    # Remove duplicate texts
    print(f"Duplicate texts: {df.duplicated(subset=['text']).sum()}")

    df = df.drop_duplicates(subset=["text"], keep="first")

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Dataset size after cleaning: {len(df)}")

    return df


def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

def preprocess_dataframe(df):
    df = clean_dataset(df)
    df["text"] = df["text"].apply(clean_text)
    return df












































