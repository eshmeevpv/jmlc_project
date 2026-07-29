"""Train-only target and numeric preprocessing from the notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REFERENCE_COLUMNS = (
    'resale_id',
    'month_index',
    'floor_area_sqm',
    'flat_type',
    'target_scaled',
)
UNSCALED_NUMERIC_COLUMNS = (
    'commercial',
    'market_hawker',
    'miscellaneous',
    'multistorey_carpark',
    'precinct_pavilion',
    'month_sin',
    'month_cos',
)
EXCLUDED_NUMERIC_COLUMNS = {
    'resale_id',
    'sale_year',
    'resale_price',
    'log_resale_price',
    'target_scaled',
    'x',
    'y',
    'hdb_ltv_1room',
    'hdb_ltv_2room',
    'hdb_ltv_3room',
    'hdb_ltv_4room',
    'hdb_ltv_5room',
    'hdb_ltv_exec',
    'hdb_ltv_others',
    *UNSCALED_NUMERIC_COLUMNS,
}


def add_month_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the exact month index and cyclic features used by the notebook."""
    result = dataframe.copy()
    if not pd.api.types.is_datetime64_any_dtype(result['transaction_date']):
        result['transaction_date'] = pd.to_datetime(result['transaction_date'], errors='coerce')
    month = result['transaction_date'].dt.month
    result['month_index'] = (result['transaction_date'].dt.year * 12 + month - 1).astype(
        'int32'
    )
    result['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12).astype('float32')
    result['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12).astype('float32')
    return result


def add_hdb_ltv_effective(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select the flat-type-specific HDB LTV series and drop source columns."""
    result = dataframe.copy()
    result['hdb_ltv_effective'] = np.select(
        [
            result['flat_type'].eq('1 ROOM'),
            result['flat_type'].eq('2 ROOM'),
            result['flat_type'].eq('3 ROOM'),
            result['flat_type'].eq('4 ROOM'),
            result['flat_type'].eq('5 ROOM'),
            result['flat_type'].eq('EXECUTIVE'),
            result['flat_type'].eq('MULTI-GENERATION'),
        ],
        [
            result['hdb_ltv_1room'],
            result['hdb_ltv_2room'],
            result['hdb_ltv_3room'],
            result['hdb_ltv_4room'],
            result['hdb_ltv_5room'],
            result['hdb_ltv_exec'],
            result['hdb_ltv_others'],
        ],
        default=np.nan,
    )
    return result.drop(
        columns=[
            'hdb_ltv_1room',
            'hdb_ltv_2room',
            'hdb_ltv_3room',
            'hdb_ltv_4room',
            'hdb_ltv_5room',
            'hdb_ltv_exec',
            'hdb_ltv_others',
        ]
    )


def prepare_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, unfitted feature construction."""
    return add_hdb_ltv_effective(add_month_features(dataframe))


@dataclass(frozen=True, slots=True)
class NumericPreprocessingArtifacts:
    """Fitted train-only transforms and feature order."""

    target_scaler: StandardScaler
    numeric_scaler: StandardScaler
    standardized_numeric_columns: tuple[str, ...]
    numerical_feature_columns: tuple[str, ...]
    coordinate_center: np.ndarray
    coordinate_scale: float
    log_transform: str = 'log'

    def transform_target(self, dataframe: pd.DataFrame) -> np.ndarray:
        return self.target_scaler.transform(dataframe[['log_resale_price']]).ravel()

    def inverse_target(self, values_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values_log = self.target_scaler.inverse_transform(
            np.asarray(values_scaled).reshape(-1, 1)
        ).reshape(-1)
        values_price = np.exp(values_log)
        return values_log, values_price

    def transform_numeric(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(
            self.numeric_scaler.transform(dataframe[list(self.standardized_numeric_columns)]),
            columns=self.standardized_numeric_columns,
            index=dataframe.index,
            dtype='float32',
        )
        transformed[['x', 'y']] = (
            (dataframe[['x', 'y']] - self.coordinate_center) / self.coordinate_scale
        ).astype('float32')
        transformed[list(UNSCALED_NUMERIC_COLUMNS)] = dataframe[
            list(UNSCALED_NUMERIC_COLUMNS)
        ].astype('float32')
        return transformed[list(self.numerical_feature_columns)]


@dataclass(frozen=True, slots=True)
class PreparedNumericSplits:
    """Prepared frames, reference values, matrices, and fitted artifacts."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_reference: pd.DataFrame
    validation_reference: pd.DataFrame
    test_reference: pd.DataFrame
    train_numeric: pd.DataFrame
    validation_numeric: pd.DataFrame
    test_numeric: pd.DataFrame
    artifacts: NumericPreprocessingArtifacts


def fit_numeric_preprocessor(
    train_dataframe: pd.DataFrame,
) -> NumericPreprocessingArtifacts:
    """Fit target, numeric, and isotropic coordinate scaling on train only."""
    target_scaler = StandardScaler().fit(train_dataframe[['log_resale_price']])
    standardized = tuple(
        column
        for column in train_dataframe.select_dtypes(include=np.number).columns
        if column not in EXCLUDED_NUMERIC_COLUMNS
    )
    numeric_scaler = StandardScaler().fit(train_dataframe[list(standardized)])
    coordinate_center = train_dataframe[['x', 'y']].mean().to_numpy(dtype=np.float64)
    coordinate_scale = float(
        np.sqrt(
            (
                (train_dataframe[['x', 'y']] - pd.Series(coordinate_center, index=['x', 'y']))
                ** 2
            )
            .to_numpy()
            .mean()
        )
    )
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('Train coordinates must have a positive finite scale.')
    feature_columns = tuple(
        column
        for column in train_dataframe.columns
        if column in (set(standardized) | set(UNSCALED_NUMERIC_COLUMNS) | {'x', 'y'})
    )
    return NumericPreprocessingArtifacts(
        target_scaler=target_scaler,
        numeric_scaler=numeric_scaler,
        standardized_numeric_columns=standardized,
        numerical_feature_columns=feature_columns,
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
    )


def prepare_numeric_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> PreparedNumericSplits:
    """Fit on train and transform all three fixed splits."""
    train_frame = prepare_feature_frame(train)
    validation_frame = prepare_feature_frame(validation)
    test_frame = prepare_feature_frame(test)
    artifacts = fit_numeric_preprocessor(train_frame)
    for frame in (train_frame, validation_frame, test_frame):
        frame['target_scaled'] = artifacts.transform_target(frame).astype('float32')
    references = tuple(
        frame[list(REFERENCE_COLUMNS)].copy()
        for frame in (train_frame, validation_frame, test_frame)
    )
    matrices = tuple(
        artifacts.transform_numeric(frame)
        for frame in (train_frame, validation_frame, test_frame)
    )
    return PreparedNumericSplits(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        train_reference=references[0],
        validation_reference=references[1],
        test_reference=references[2],
        train_numeric=matrices[0],
        validation_numeric=matrices[1],
        test_numeric=matrices[2],
        artifacts=artifacts,
    )
