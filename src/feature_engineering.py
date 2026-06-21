import re
import numpy as np


def extract_stylometry(df):
    # Ratio of fully capitalized words
    df["caps_ratio"] = df["text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
        / (len(str(x).split()) + 1)
    )

    # Number of exclamation marks
    df["exclamation_count"] = df["text"].apply(
        lambda x: str(x).count("!")
    )

    # Number of question marks
    df["question_count"] = df["text"].apply(
        lambda x: str(x).count("?")
    )

    # Original text length
    df["text_len"] = df["text"].apply(
        lambda x: len(str(x))
    )

    return df


def extract_text_statistics(df):
    # Character length
    df["text_length"] = df["clean_text"].apply(len)

    # Number of words
    df["word_count"] = df["clean_text"].apply(
        lambda x: len(x.split())
    )

    # Average word length
    df["avg_word_length"] = df["clean_text"].apply(
        lambda x: (
            np.mean([len(word) for word in x.split()])
            if len(x.split()) > 0
            else 0
        )
    )

    # Number of numeric expressions in original text
    df["number_count"] = df["text"].apply(
        lambda x: len(re.findall(r"\d+", str(x)))
    )

    return df


def engineer_features(df):
    df = extract_stylometry(df)
    df = extract_text_statistics(df)

    return df
