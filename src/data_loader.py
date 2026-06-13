"""Utilities for downloading Trump Tweets dataset from Kaggle."""

from __future__ import annotations

import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def _resolve_kaggle_config_dir() -> Path:
    """Return the expected Kaggle config directory."""
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))


def _has_kaggle_credentials() -> bool:
    """Return True if any supported Kaggle credential source is available."""
    config_dir = _resolve_kaggle_config_dir()
    kaggle_json = config_dir / "kaggle.json"
    access_token = config_dir / "access_token"

    has_legacy_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(
        os.environ.get("KAGGLE_KEY")
    )
    has_token_env = bool(os.environ.get("KAGGLE_API_TOKEN"))

    return (
        kaggle_json.exists()
        or access_token.exists()
        or has_legacy_env
        or has_token_env
    )


def download_trump_tweets(data_dir: str | Path = "data") -> Path:
    """Download and unzip the Kaggle dataset, then return CSV file path.

    Args:
        data_dir: Directory where the dataset files should be stored.

    Returns:
        Path to the first CSV file extracted from the dataset.

    Raises:
        FileNotFoundError: If Kaggle API credentials are missing or CSV is absent.
    """
    if not _has_kaggle_credentials():
        config_dir = _resolve_kaggle_config_dir()
        raise FileNotFoundError(
            "Missing Kaggle credentials. Provide one of: "
            f"{config_dir / 'kaggle.json'}, {config_dir / 'access_token'}, "
            "or environment variables KAGGLE_API_TOKEN / KAGGLE_USERNAME + KAGGLE_KEY."
        )

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "austinreese/trump-tweets",
        path=str(destination),
        unzip=True,
        quiet=False,
    )

    csv_files = sorted(destination.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {destination} after download from Kaggle."
        )
    return csv_files[0]
