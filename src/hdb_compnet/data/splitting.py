"""Train/validation/test split"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_hdb_transactions(
    dataframe: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return 70/10/20 split, stratified by sale year"""
    data = dataframe.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['transaction_date']):
        data['transaction_date'] = pd.to_datetime(data['transaction_date'], errors='coerce')
    if 'sale_year' not in data:
        data['sale_year'] = data['transaction_date'].dt.year

    train, temporary = train_test_split(
        data,
        test_size=0.30,
        random_state=random_state,
        shuffle=True,
        stratify=data['sale_year'],
    )
    validation, test = train_test_split(
        temporary,
        test_size=2 / 3,
        random_state=random_state,
        shuffle=True,
        stratify=temporary['sale_year'],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
