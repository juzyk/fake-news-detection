import numpy as np

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_svm_model(X_train_tfidf, y_train):

    svm_model = LinearSVC(
        C=0.7847599703514607,
        loss="squared_hinge",
        class_weight=None
    )

    svm_calibrated = CalibratedClassifierCV(
        svm_model,
        cv=3
    )

    svm_calibrated.fit(
        X_train_tfidf,
        y_train
    )

    return svm_calibrated


def train_xgboost_model(X_train_tfidf, y_train):

    xgb_model = XGBClassifier(
        subsample=1.0,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.3,
        gamma=0,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    xgb_model.fit(
        X_train_tfidf,
        y_train
    )

    return xgb_model


def ensemble_predict(
    svm_model,
    xgb_model,
    X_test_tfidf,
    threshold=0.5
):

    svm_proba = (
        svm_model
        .predict_proba(X_test_tfidf)[:, 1]
    )

    xgb_proba = (
        xgb_model
        .predict_proba(X_test_tfidf)[:, 1]
    )

    final_proba = (
        svm_proba + xgb_proba
    ) / 2

    predictions = (
        final_proba >= threshold
    ).astype(int)

    return predictions, final_proba


def evaluate_ensemble(
    y_true,
    y_pred
):

    results = {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred
        ),

        "Recall": recall_score(
            y_true,
            y_pred
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred
        )
    }

    return results


def train_ensemble(
    X_train_tfidf,
    y_train
):

    svm_model = train_svm_model(
        X_train_tfidf,
        y_train
    )

    xgb_model = train_xgboost_model(
        X_train_tfidf,
        y_train
    )

    return svm_model, xgb_model
