"""Data loading, pseudo-labeling, preprocessing, and TF-IDF vectorization."""

from __future__ import annotations
import re
from pathlib import Path
import nltk
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder

emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=0)

def _ensure_nltk_resources() -> None:
    """Pobiera zasoby NLTK."""
    nltk.download("stopwords", quiet=True)

def _clean_text(text: str, stop_words: set[str]) -> str:
    """Usuwa linki, znaki interpunkcyjne i nudne słowa."""
    no_urls = re.sub(r"https?://\S+|www\.\S+", " ", text)
    only_letters = re.sub(r"[^a-zA-Z\s]", " ", no_urls)
    normalized = only_letters.lower()
    tokens = [token for token in normalized.split() if token and token not in stop_words]
    return " ".join(tokens)


def _pseudo_label_tweet(text: str) -> str:
    """Wykrywa dominującą emocję w tweecie."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutral"
    text = text[:512]
    result = emotion_analyzer(text)
    return result[0][0]['label']


def prepare_dataset(csv_path: str | Path, max_features: int = 5000) -> tuple[spmatrix, pd.Series, TfidfVectorizer, pd.DataFrame, str, LabelEncoder]:
    """Przygotowuje w pełni gotowy zbiór do uczenia maszynowego."""
    _ensure_nltk_resources()

    dataframe = pd.read_csv(csv_path)
    # dataframe = dataframe.sample(n=2000, random_state=42).copy()
    text_column = "content"
    stop_words = set(stopwords.words("english"))
    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)

    print("Czyszczenie tekstu...")
    dataframe["clean_text"] = dataframe[text_column].apply(lambda text: _clean_text(text, stop_words))
    
    print("\nOcenianie emocji tweetów (to potrwa kilka minut)...")
    dataframe["sentiment_label"] = dataframe[text_column].apply(lambda text: _pseudo_label_tweet(text))

    label_encoder = LabelEncoder()
    dataframe["encoded_label"] = label_encoder.fit_transform(dataframe["sentiment_label"])

    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(dataframe["clean_text"])
    
    labels = dataframe["encoded_label"]

    return features, labels, vectorizer, dataframe, text_column, label_encoder