"""Explicit Kaggle download helpers with no import-time side effects."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEFAULT_DATASET_SLUG = 'paveleshmeev/sgp-resale-prices-90'
DEFAULT_DATA_FILENAME = 'df_common_sample_complete_case.csv'
DEFAULT_CACHE_DATASET_SLUG = 'paveleshmeev/hdb-compnet-cache'


def _validate_kaggle_credentials() -> None:
    credentials_file = Path.home() / '.kaggle' / 'kaggle.json'
    has_environment = bool(os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'))
    if not has_environment and not credentials_file.is_file():
        raise RuntimeError(
            'Kaggle credentials are missing. Set KAGGLE_USERNAME and KAGGLE_KEY '
            'or configure ~/.kaggle/kaggle.json.'
        )


def download_kaggle_dataset_file(
    destination: str | Path,
    *,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    filename: str = DEFAULT_DATA_FILENAME,
    force: bool = False,
) -> Path:
    """Download one Kaggle dataset file into ``destination``."""
    _validate_kaggle_credentials()
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    output_path = destination_path / filename
    if output_path.exists() and not force:
        return output_path
    command = [
        'kaggle',
        'datasets',
        'download',
        '--dataset',
        dataset_slug,
        '--file',
        filename,
        '--path',
        str(destination_path),
        '--unzip',
    ]
    if force:
        command.append('--force')
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "The 'kaggle' executable is unavailable; install the download extra."
        ) from error
    return output_path


def download_kaggle_cache_artifacts(
    destination: str | Path,
    *,
    dataset_slug: str = DEFAULT_CACHE_DATASET_SLUG,
    force: bool = False,
) -> Path:
    """Download and unzip the notebook's external retrieval-cache dataset."""
    _validate_kaggle_credentials()
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    command = [
        'kaggle',
        'datasets',
        'download',
        '--dataset',
        dataset_slug,
        '--path',
        str(destination_path),
        '--unzip',
        '--quiet',
    ]
    if force:
        command.append('--force')
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "The 'kaggle' executable is unavailable; install the download extra."
        ) from error
    return destination_path
