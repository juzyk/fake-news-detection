from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import pandas as pd


def prepare_data(df):
    X = df["clean_text"].astype(str)
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_vectorizer(X_train, X_test):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    }


def train_models(X_train_tfidf, y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Naive Bayes": MultinomialNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "SVM": LinearSVC(),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model

    return trained_models


def compare_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

    return pd.DataFrame(results)


def train_pipeline(df):

    X_train, X_test, y_train, y_test = prepare_data(df)

    vectorizer, X_train_tfidf, X_test_tfidf = build_vectorizer(
        X_train,
        X_test
    )

    models = train_models(
        X_train_tfidf,
        y_train
    )

    comparison_df = compare_models(
        models,
        X_test_tfidf,
        y_test
    )

    return (
        models,
        vectorizer,
        comparison_df
    )
