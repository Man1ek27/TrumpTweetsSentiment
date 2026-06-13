"""Utilities for downloading Trump Tweets dataset from Kaggle."""

from __future__ import annotations

import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def _resolve_kaggle_config_path() -> Path:
    """Return the expected Kaggle API key path."""
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return config_dir / "kaggle.json"


def download_trump_tweets(data_dir: str | Path = "data") -> Path:
    """Download and unzip the Kaggle dataset, then return CSV file path.

    Args:
        data_dir: Directory where the dataset files should be stored.

    Returns:
        Path to the first CSV file extracted from the dataset.

    Raises:
        FileNotFoundError: If Kaggle API credentials are missing or CSV is absent.
    """
    kaggle_config = _resolve_kaggle_config_path()
    if not kaggle_config.exists():
        raise FileNotFoundError(
            "Missing Kaggle credentials. Expected file: "
            f"{kaggle_config}. Place your kaggle.json there."
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
