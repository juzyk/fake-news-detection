import pandas as pd
import joblib


def analyze_disagreements(
    y_test,
    svm_model,
    xgb_model,
    X_test_tfidf
):

    comparison = pd.DataFrame({
        "y_true": y_test,
        "svm_pred": svm_model.predict(X_test_tfidf),
        "xgb_pred": xgb_model.predict(X_test_tfidf)
    })

    comparison["agree"] = (
        comparison["svm_pred"]
        == comparison["xgb_pred"]
    )

    different = comparison[
        comparison["svm_pred"]
        != comparison["xgb_pred"]
    ]

    print(
        "Number of disagreements:",
        len(different)
    )

    print(
        "Total samples:",
        len(comparison)
    )

    print(
        "Disagreement rate:",
        len(different) / len(comparison)
    )

    xgb_correct = different[
        different["xgb_pred"]
        == different["y_true"]
    ]

    svm_correct = different[
        different["svm_pred"]
        == different["y_true"]
    ]

    xgb_correct_pct = (
        len(xgb_correct)
        / len(different)
    ) * 100

    svm_correct_pct = (
        len(svm_correct)
        / len(different)
    ) * 100

    print("\nDisagreement Analysis")

    print(
        f"XGBoost correct in disagreements: "
        f"{xgb_correct_pct:.2f}%"
    )

    print(
        f"SVM correct in disagreements: "
        f"{svm_correct_pct:.2f}%"
    )

    print("\nRaw counts:")

    print(
        "XGBoost correct:",
        len(xgb_correct)
    )

    print(
        "SVM correct:",
        len(svm_correct)
    )

    return comparison


def save_models(
    svm_model,
    xgb_model,
    vectorizer
):

    ensemble_model = {
        "svm": svm_model,
        "xgb": xgb_model,
        "svm_weight": 0.4,
        "xgb_weight": 0.6
    }

    joblib.dump(
        ensemble_model,
        "model.pkl"
    )

    joblib.dump(
        vectorizer,
        "vectorizer.pkl"
    )

    print(
        "Models, weights, and vectorizer "
        "saved successfully!"
    )
