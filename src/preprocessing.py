import pandas as pd
import re
import string
import unicodedata
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download once if needed
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_dataset(df):
    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str)

    df = df[df["text"].str.strip() != ""]

    print(f"Duplicate texts: {df.duplicated(subset=['text']).sum()}")

    df = df.drop_duplicates(subset=["text"], keep="first")

    df = df.reset_index(drop=True)

    print(f"Dataset size after cleaning: {len(df)}")

    return df


def absolute_cleaning_pipeline(text):
    text = (
        unicodedata
        .normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    # Remove URLs and HTML
    text = re.sub(r"https?://\S+|www\.\S+|<.*?>+", "", text)

    # Remove Reuters/CNN/BBC prefixes
    text = re.sub(
        r"^.*?\(reuters\)\s*[-—]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^.*?\(cnn\)\s*[-—]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^.*?\(bbc\)\s*[-—]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.lower()

    # Remove punctuation
    text = re.sub(
        "[%s]" % re.escape(string.punctuation),
        "",
        text
    )

    # Remove words containing digits
    text = re.sub(r"\w*\d\w*", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal + lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_tokens)


def remove_rare_words(df, min_frequency=5):
    word_freq = Counter()

    for text in df["clean_text"]:
        word_freq.update(str(text).split())

    rare_words = {
        word
        for word, count in word_freq.items()
        if count <= min_frequency
    }

    df["clean_text"] = df["clean_text"].apply(
        lambda x: " ".join(
            word
            for word in str(x).split()
            if word not in rare_words
        )
    )

    print(f"Rare words removed: {len(rare_words)}")

    return df


def preprocess_dataframe(df):
    df = clean_dataset(df)

    df["clean_text"] = df["text"].apply(
        absolute_cleaning_pipeline
    )

    df = remove_rare_words(df)

    return df












































