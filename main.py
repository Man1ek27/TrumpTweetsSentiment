"""Główny punkt analizy sentymentu tweetów Trumpa i wyjaśnialności SHAP."""

from src.data_loader import download_trump_tweets
from src.data_processing import prepare_dataset
from src.model_training import train_and_evaluate_model
from src.shap_analysis import generate_shap_visualizations
from src.eda import plot_length, plot_emotion_distribution, plot_top_words, plot_confusion_matrix


def main() -> None:
    """Uruchamia pełny potok: pobieranie -> przetwarzanie -> turniej modeli -> analiza SHAP."""

    csv_path = download_trump_tweets(data_dir="data")
    
    features, labels, vectorizer, dataframe, text_column, label_encoder = prepare_dataset(csv_path)
    
    print("Generowanie wykresów...")
    plot_length(csv_path, results_dir="results")
    plot_top_words(dataframe)
    plot_emotion_distribution(dataframe, results_dir="results")
    
    print("\n--- Trening modeli ---")
    artifacts = train_and_evaluate_model(features, labels, label_encoder)
    plot_confusion_matrix(model=artifacts.model, x_test=artifacts.x_test, y_test=artifacts.y_test, label_encoder=label_encoder, results_dir="results")
    
    print("\n--- Generowanie wykresów SHAP dla najlepszego modelu ---")
    generate_shap_visualizations(
        model=artifacts.model,
        x_train=artifacts.x_train,
        x_test=artifacts.x_test,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        results_dir="results",
    )

    print("Pipeline zakończony. Wykresy SHAP zostały zapisane w katalogu /results/.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline nie powiódł się: {exc}")
        raise SystemExit(1) from exc
