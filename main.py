"""Main entrypoint for Trump Tweets sentiment analysis and SHAP explainability."""

from __future__ import annotations

from src.data_loader import download_trump_tweets
from src.data_processing import prepare_dataset
from src.model_training import train_and_evaluate_model
from src.shap_analysis import generate_shap_visualizations


def main() -> None:
    """Run full pipeline: download -> process -> train -> evaluate -> explain."""
    csv_path = download_trump_tweets(data_dir="data")
    features, labels, vectorizer, _, _ = prepare_dataset(csv_path)
    artifacts = train_and_evaluate_model(features, labels)
    generate_shap_visualizations(
        model=artifacts.model,
        x_train=artifacts.x_train,
        x_test=artifacts.x_test,
        vectorizer=vectorizer,
        results_dir="results",
    )
    print("Pipeline finished. SHAP plots saved in the results/ directory.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - entrypoint error handling
        print(f"Pipeline failed: {exc}")
        raise SystemExit(1) from exc
