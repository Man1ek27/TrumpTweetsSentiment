"""Data loading, pseudo-labeling, preprocessing, and TF-IDF vectorization."""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources when missing."""
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)


def _detect_text_column(dataframe: pd.DataFrame) -> str:
    """Infer tweet text column from common names or fallback to first object column."""
    candidates = ["content", "text", "tweet", "tweet_text", "body"]
    existing = [column for column in candidates if column in dataframe.columns]
    if existing:
        return existing[0]

    object_columns = dataframe.select_dtypes(include=["object"]).columns.tolist()
    if not object_columns:
        raise ValueError("No textual column found in input CSV.")
    return object_columns[0]


def _clean_text(text: str, stop_words: set[str]) -> str:
    """Normalize text by removing URLs, punctuation and stop words."""
    no_urls = re.sub(r"https?://\S+|www\.\S+", " ", text)
    only_letters = re.sub(r"[^a-zA-Z\s]", " ", no_urls)
    normalized = only_letters.lower()
    tokens = [token for token in normalized.split() if token and token not in stop_words]
    return " ".join(tokens)


def _pseudo_label_tweet(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
    """Create pseudo sentiment labels using VADER compound score."""
    compound = analyzer.polarity_scores(text)["compound"]
    if compound > POSITIVE_THRESHOLD:
        return "positive"
    if compound < NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def prepare_dataset(
    csv_path: str | Path,
    max_features: int = 5000,
) -> tuple[spmatrix, pd.Series, TfidfVectorizer, pd.DataFrame, str]:
    """Load CSV, generate pseudo-labels, clean text and vectorize with TF-IDF.

    Args:
        csv_path: Input CSV path.
        max_features: Maximum TF-IDF vocabulary size.

    Returns:
        Tuple with: vectorized features, labels, vectorizer, processed dataframe, text column.
    """
    _ensure_nltk_resources()

    dataframe = pd.read_csv(csv_path)
    text_column = _detect_text_column(dataframe)
    stop_words = set(stopwords.words("english"))
    sentiment_analyzer = SentimentIntensityAnalyzer()

    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    dataframe["clean_text"] = dataframe[text_column].apply(
        lambda text: _clean_text(text, stop_words)
    )
    dataframe["sentiment_label"] = dataframe[text_column].apply(
        lambda text: _pseudo_label_tweet(text, sentiment_analyzer)
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(dataframe["clean_text"])
    labels = dataframe["sentiment_label"]

    return features, labels, vectorizer, dataframe, text_column
