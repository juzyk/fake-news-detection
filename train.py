from src.data_loader import load_all_data
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import engineer_features

from src.train_model import (
    prepare_data,
    build_vectorizer
)

from src.hyperparameter_tuning import (
    tune_svm,
    tune_xgboost
)

from src.ensemble import (
    train_xgboost_model,
    ensemble_predict,
    evaluate_ensemble
)

from src.save_model import (
    analyze_disagreements,
    save_models
)


def main():

    print("Loading datasets...")
    df = load_all_data()

    print("Preprocessing data...")
    df = preprocess_dataframe(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Preparing train-test split...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Building TF-IDF vectorizer...")
    vectorizer, X_train_tfidf, X_test_tfidf = build_vectorizer(
        X_train,
        X_test
    )

    print("Tuning SVM...")
    svm_search = tune_svm(
        X_train_tfidf,
        y_train
    )

    print("\nBest SVM Parameters:")
    print(svm_search.best_params_)

    svm_model = svm_search.best_estimator_

    print("Tuning XGBoost...")
    xgb_search = tune_xgboost(
        X_train_tfidf,
        y_train
    )

    print("\nBest XGBoost Parameters:")
    print(xgb_search.best_params_)

    print("Training final XGBoost...")
    xgb_model = train_xgboost_model(
        X_train_tfidf,
        y_train
    )

    print("Running ensemble...")
    y_pred, probabilities = ensemble_predict(
        svm_model,
        xgb_model,
        X_test_tfidf
    )

    results = evaluate_ensemble(
        y_test,
        y_pred
    )

    print("\n Ensemble results")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nAnalyzing disagreements...")

    analyze_disagreements(
        y_test,
        svm_model,
        xgb_model,
        X_test_tfidf
    )

    print("\nSaving models...")

    save_models(
        svm_model,
        xgb_model,
        vectorizer
    )

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
