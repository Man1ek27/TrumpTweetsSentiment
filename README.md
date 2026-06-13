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

Kaggle udostepnia obecnie dwa sposoby autoryzacji. Mozesz uzyc dowolnego z nich:

1. Nowy token API (zalecane):
    - Ustaw zmienna srodowiskowa `KAGGLE_API_TOKEN`, albo
    - zapisz token w pliku:
      - Linux/macOS: `~/.kaggle/access_token`
      - Windows: `%USERPROFILE%\.kaggle\access_token`

2. Klasyczny plik `kaggle.json`:
    - Linux/macOS: `~/.kaggle/kaggle.json`
    - Windows: `%USERPROFILE%\.kaggle\kaggle.json`

Przyklad dla Windows PowerShell (token z ekranu Kaggle):

```powershell
$env:KAGGLE_API_TOKEN = "WSTAW_TUTAJ_TOKEN"
setx KAGGLE_API_TOKEN "WSTAW_TUTAJ_TOKEN"
```

Ustaw prawa dostepu (Linux/macOS):

```bash
chmod 600 ~/.kaggle/kaggle.json
# lub, jesli uzywasz nowego tokenu:
chmod 600 ~/.kaggle/access_token
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


## Użyty Prompt Startowy

```text
Jako Ekspert Data Science i Inżynier Oprogramowania, proszę o wygenerowanie kompletnego, modularnego projektu w języku Python, realizującego zadanie analizy sentymentu NLP oraz wyjaśnialności modelu (XAI) za pomocą biblioteki SHAP.

Kontekst i cel:
Projekt polega na zbudowaniu modelu klasyfikującego sentyment tweetów ze zbioru "Trump Tweets" (Kaggle) oraz przeprowadzeniu szczegółowej analizy wpływu poszczególnych słów na predykcję za pomocą SHAP. Zbiór danych nie posiada gotowych etykiet sentymentu, dlatego konieczne jest ich wcześniejsze wygenerowanie.

Wymagania architektoniczne (struktura plików):
Proszę stworzyć czyste, dobrze udokumentowane pliki .py (z użyciem docstringów i type hintingu). Wymagana struktura to:

- requirements.txt - lista zależności (w tym: pandas, scikit-learn, shap, nltk, matplotlib, kaggle)
- README.md - instrukcja instalacji, uruchomienia projektu oraz konfiguracji klucza API Kaggle (kaggle.json), który jest wymagany do pobrania danych
- src/data_loader.py - moduł automatycznie pobierający zbiór danych z Kaggle (austinreese/trump-tweets) za pomocą pakietu kaggle i rozpakowujący plik CSV do folderu data/
- src/data_processing.py - moduł odpowiedzialny za:
    - wczytanie danych CSV
    - pseudo-labeling: wykorzystanie analizatora SentimentIntensityAnalyzer z biblioteki nltk.sentiment.vader do oceny każdego tweeta i stworzenia nowej kolumny celu (np. sentiment_label: pozytywny, negatywny, neutralny na podstawie wartości compound)
    - czyszczenie tekstu (usuwanie URLi, znaków specjalnych, stop words)
    - wektoryzację tekstu przy użyciu TfidfVectorizer z scikit-learn
- src/model_training.py - moduł do podziału danych na zbiór uczący i testowy, trenowania klasyfikatora (proszę użyć LogisticRegression lub RandomForestClassifier) oraz ewaluacji (generowanie i wypisywanie w konsoli pełnego Raportu Klasyfikacji i Macierzy Pomyłek)
- src/shap_analysis.py - moduł wykorzystujący bibliotekę shap. Musi obliczyć wartości SHAP i wygenerować wizualizacje (np. summary_plot oraz waterfall_plot dla wybranej próbki). Wykresy mają być automatycznie zapisywane w nowym folderze results/ jako pliki .png (bez blokowania działania skryptu przez plt.show())
- main.py - główny skrypt orkiestrujący, który chronologicznie wywołuje powyższe moduły

Wytyczne techniczne:
Rozwiązanie ma w całości opierać się na klasycznym uczeniu maszynowym (sklearn) oraz NLP (nltk). Proszę zadbać o czytelność kodu, odpowiednią obsługę błędów (np. sprawdzenie obecności pliku .kaggle/kaggle.json) i pełną automatyzację procesu od pobrania danych po zapis wykresów.
```