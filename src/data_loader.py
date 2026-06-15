"""Pobieranie datasetu Tramp Tweets z Kaggle."""

import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def _resolve_kaggle_config_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))

def _has_kaggle_credentials() -> bool:
    config_dir = _resolve_kaggle_config_dir()
    kaggle_json = config_dir / "kaggle.json"
    access_token = config_dir / "access_token"
    has_legacy_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    has_token_env = bool(os.environ.get("KAGGLE_API_TOKEN"))
    return (kaggle_json.exists() or access_token.exists() or has_legacy_env or has_token_env)

def download_trump_tweets(data_dir: str | Path = "data") -> Path:
    """Pobiera i rozpakowuje dane z Kaggle."""
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(destination.glob("*.csv"))

    if csv_files:
        print(f"Plik CSV już istnieje w folderze '{destination}'. Pomijam pobieranie.")
        return csv_files[0]

    if not _has_kaggle_credentials():
        config_dir = _resolve_kaggle_config_dir()
        raise FileNotFoundError(f"Missing Kaggle credentials. Uzupełnij plik kaggle.json w {config_dir}")

    print("Brak plików na dysku. Rozpoczynam pobieranie z Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("austinreese/trump-tweets", path=str(destination), unzip=True, quiet=False)

    csv_files = sorted(destination.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nie znaleziono plików CSV po pobraniu.")
    return csv_files[0]