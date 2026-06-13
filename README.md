# TrumpTweetsSentiment

Modularny projekt NLP do analizy sentymentu tweetów Donalda Trumpa z automatycznym pseudo-labelingiem oraz wyjaśnialnością modelu (XAI) z użyciem SHAP.

## Wymagania

- Python 3.10+
- Konto Kaggle i klucz API (`kaggle.json`)

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Konfiguracja Kaggle API

1. Pobierz `kaggle.json` z ustawień konta Kaggle: https://www.kaggle.com/settings
2. Umieść plik w:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
3. Ustaw prawa dostępu (Linux/macOS):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Alternatywnie ustaw zmienną środowiskową `KAGGLE_CONFIG_DIR` wskazującą na katalog z `kaggle.json`.

## Uruchomienie

```bash
python main.py
```

Pipeline automatycznie:

1. Pobiera i rozpakowuje zbiór `austinreese/trump-tweets` do `data/`.
2. Tworzy pseudo-etykiety sentymentu (VADER: positive / neutral / negative).
3. Czyści tekst i wektoryzuje dane przez TF-IDF.
4. Trenuje model `LogisticRegression`.
5. Wypisuje raport klasyfikacji i macierz pomyłek.
6. Generuje wykresy SHAP i zapisuje je do `results/` jako pliki `.png`.

## Struktura projektu

```text
.
├── main.py
├── requirements.txt
├── README.md
└── src
    ├── data_loader.py
    ├── data_processing.py
    ├── model_training.py
    └── shap_analysis.py
```