import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def tune_xgboost(X_train_tfidf, y_train):

    xgb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.3]
    }

    xgb_random = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ),
        param_distributions=xgb_params,
        n_iter=5,
        scoring="f1",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=1
    )

    xgb_random.fit(X_train_tfidf, y_train)

    return xgb_random


def tune_svm(X_train_tfidf, y_train):

    svm_params = {
        "C": np.logspace(-3, 2, 20),
        "loss": ["hinge", "squared_hinge"],
        "class_weight": [None, "balanced"]
    }

    svm_random = RandomizedSearchCV(
        LinearSVC(),
        param_distributions=svm_params,
        n_iter=30,
        cv=5,
        scoring="f1",
        random_state=42,
        verbose=2,
        n_jobs=-1
    )

    svm_random.fit(X_train_tfidf, y_train)

    return svm_random


def evaluate_tuned_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    }

    return results
