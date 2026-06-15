"""Analiza danych"""

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix

def plot_length(csv_path: str, results_dir: str = "results") -> None:
    """Generuje podstawowe statystyki i wykresy dla surowych danych."""
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    text_column = "content"

    print(f"\nLiczba wszystkich tweetów w bazie wejściowej: {df.shape[0]}")
    df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, color='#2a788e', kde=True)
    plt.title('Rozkład liczby słów w tweetach Trumpa', fontsize=16)
    plt.xlabel('Liczba słów', fontsize=12)
    plt.ylabel('Liczba tweetów', fontsize=12)
    plt.xlim(0, 60)
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(results_dir, 'word_count_dist.png'), bbox_inches='tight')
    plt.show()

def plot_top_words(dataframe: pd.DataFrame, column_name: str = "clean_text", top_n: int = 20, results_dir: str = "results") -> None:
    """Generuje wykres słupkowy najczęściej występujących słów na przetworzonym tekście."""
    os.makedirs(results_dir, exist_ok=True)
    all_words = " ".join(dataframe[column_name].dropna().astype(str)).split()
    word_counts = Counter(all_words)
    words_df = pd.DataFrame(word_counts.most_common(top_n), columns=['Słowo', 'Liczba wystąpień'])

    plt.figure(figsize=(12, 8))
    sns.barplot(data=words_df, x='Liczba wystąpień', y='Słowo', hue='Słowo', palette='viridis', legend=False)
    plt.title(f'{top_n} najczęściej używanych słów przez Trumpa', fontsize=16)
    plt.xlabel('Liczba wystąpień', fontsize=12)
    plt.ylabel('Słowo', fontsize=12)
    plt.grid(axis='x', alpha=0.3)

    plt.savefig(os.path.join(results_dir, 'top_words.png'), bbox_inches='tight')
    plt.show()

def plot_emotion_distribution(dataframe: pd.DataFrame, results_dir: str = "results") -> None:
    """Generuje wykres pokazujący podział tweetów na konkretne emocje."""
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    ax = sns.countplot(
        data=dataframe, x='sentiment_label', hue='sentiment_label',
        palette='viridis', order=dataframe['sentiment_label'].value_counts().index, legend=False
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)

    plt.title('Rozkład emocji w analizowanych tweetach Trumpa', fontsize=16)
    plt.xlabel('Wykryta emocja', fontsize=12)
    plt.ylabel('Liczba tweetów', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(os.path.join(results_dir, 'emotion_distribution.png'), bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(model, x_test, y_test, label_encoder, results_dir="results"):
    """Generuje i zapisuje macierz pomyłek."""
    predictions = model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Macierz pomyłek (Regresja Logistyczna)")
    plt.xlabel("Przewidywana emocja")
    plt.ylabel("Rzeczywista emocja")
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.close()
    print(f"Macierz pomyłek zapisana w {results_dir}/confusion_matrix.png")