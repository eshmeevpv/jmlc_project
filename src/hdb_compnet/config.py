"""Notebook-derived configuration for HDB-CompNet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from hdb_compnet.utils.serialization import convert_to_serializable

BRANCH_NAMES = ('baseline', 'structural', 'spatial', 'market')
RETRIEVAL_SPACE_NAMES = ('structural', 'spatial', 'market')
CATEGORY_COLUMNS = (
    'town',
    'flat_type',
    'flat_model',
    'nearest_hospital_type',
    'nearest_hospital_category',
)
RETRIEVAL_CATEGORICAL_COLUMNS = {
    'structural': ('flat_type', 'flat_model'),
    'spatial': (
        'flat_type',
        'town',
        'nearest_hospital_type',
        'nearest_hospital_category',
    ),
    'market': ('flat_type',),
}
RETRIEVAL_SPACES = {
    'structural': {
        'numeric': (
            'max_floor_lvl',
            'year_completed',
            'commercial',
            'market_hawker',
            'miscellaneous',
            'multistorey_carpark',
            'precinct_pavilion',
            'total_dwelling_units',
            'floor_mean',
            'floor_area_sqm',
            'lease_commence_date',
            'remaining_lease_full_years',
        ),
        'categorical': ('flat_model',),
    },
    'spatial': {
        'numeric': (
            'x',
            'y',
            'dist_mrt_exit_m',
            'mrt_station_count_1km',
            'walk_time_to_nearest_mrt_exit_min',
            'dist_to_raffles_place_m',
            'dist_nearest_primary_school_m',
            'primary_school_count_1km',
            'dist_nearest_secondary_school_m',
            'secondary_school_count_1km',
            'dist_nearest_park_m',
            'park_count_1km',
            'dist_nearest_hospital_m',
            'hospital_count_1km',
            'dist_nearest_hawker_centre_m',
            'hawker_centre_count_1km',
        ),
        'categorical': (
            'town',
            'nearest_hospital_type',
            'nearest_hospital_category',
        ),
    },
    'market': {
        'numeric': (
            'month_index',
            'month_sin',
            'month_cos',
            'bank_hdb_max_tenure_years',
            'bank_hdb_msr',
            'hdb_housing_loan_max_tenure_years',
            'hdb_housing_msr',
            'ltv_first_std',
            'ltv_second_std',
            'ltv_third_plus_std',
            'ltv_non_individual_std',
            'ltv_first_long',
            'ltv_second_long',
            'ltv_third_plus_long',
            'ltv_any_existing_std',
            'ltv_any_existing_long',
            'cpf_oa_rate',
            'hdb_ltv_effective',
            'cpi_all_items',
            'sgs_2y_yield',
            'real_gdp_sa',
            'resident_population',
            'resident_unemployment_rate',
            'ulc_index_sa',
            'per_capita_gdp_real',
            'resident_households_hdb',
            'household_net_worth',
        ),
        'categorical': (),
    },
}


def _immutable_int_mapping(values: Mapping[str, Any]) -> Mapping[str, int]:
    return MappingProxyType({str(key): int(value) for key, value in values.items()})


def _immutable_index_mapping(
    values: Mapping[str, Any],
) -> Mapping[str, tuple[int, ...]]:
    return MappingProxyType(
        {str(key): tuple(int(index) for index in value) for key, value in values.items()}
    )


@dataclass(frozen=True, slots=True)
class HDBCompNetConfig:
    """Complete model and training configuration used by the notebook."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_cardinalities: Mapping[str, int]
    embedding_dimensions: Mapping[str, int]
    numeric_column_indices: Mapping[str, int]
    categorical_column_indices: Mapping[str, int]
    retrieval_numeric_indices: Mapping[str, tuple[int, ...]]
    retrieval_categorical_indices: Mapping[str, tuple[int, ...]]
    branch_names: tuple[str, ...] = BRANCH_NAMES
    retrieval_space_names: tuple[str, ...] = RETRIEVAL_SPACE_NAMES
    global_hidden_dim: int = 256
    global_embedding_dim: int = 128
    retrieval_hidden_dim: int = 128
    retrieval_embedding_dim: int = 64
    score_hidden_dim: int = 128
    gate_hidden_dim: int = 64
    reliability_feature_dim: int = 5
    dropout: float = 0.1
    k_search: int = 128
    k_candidates: int = 32
    retrieval_query_batch_size: int = 1024
    candidate_filter_batch_size: int = 8192
    huber_delta: float = 1.0
    auxiliary_loss_weight: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 30
    gradient_clip_norm: float | None = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 5
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    use_amp: bool = True
    amp_dtype: str = 'float16'
    seed: int = 42
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    test_fraction: float = 0.2
    stratify_column: str = 'sale_year'
    log_transform: str = 'log'
    dataset_slug: str = 'paveleshmeev/sgp-resale-prices-90'
    dataset_filename: str = 'df_common_sample_complete_case.csv'
    cache_dataset_slug: str = 'paveleshmeev/hdb-compnet-cache'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'numeric_columns', tuple(self.numeric_columns))
        object.__setattr__(self, 'categorical_columns', tuple(self.categorical_columns))
        object.__setattr__(self, 'branch_names', tuple(self.branch_names))
        object.__setattr__(self, 'retrieval_space_names', tuple(self.retrieval_space_names))
        object.__setattr__(
            self, 'category_cardinalities', _immutable_int_mapping(self.category_cardinalities)
        )
        object.__setattr__(
            self, 'embedding_dimensions', _immutable_int_mapping(self.embedding_dimensions)
        )
        object.__setattr__(
            self, 'numeric_column_indices', _immutable_int_mapping(self.numeric_column_indices)
        )
        object.__setattr__(
            self,
            'categorical_column_indices',
            _immutable_int_mapping(self.categorical_column_indices),
        )
        object.__setattr__(
            self,
            'retrieval_numeric_indices',
            _immutable_index_mapping(self.retrieval_numeric_indices),
        )
        object.__setattr__(
            self,
            'retrieval_categorical_indices',
            _immutable_index_mapping(self.retrieval_categorical_indices),
        )
        self.validate()

    @property
    def total_embedding_dim(self) -> int:
        return sum(self.embedding_dimensions[column] for column in self.categorical_columns)

    @property
    def global_input_dim(self) -> int:
        return len(self.numeric_columns) + self.total_embedding_dim

    @property
    def pair_feature_dim(self) -> int:
        return 5 * self.retrieval_embedding_dim + 1

    @property
    def gate_input_dim(self) -> int:
        return (
            self.global_embedding_dim
            + len(self.branch_names)
            + len(self.retrieval_space_names) * self.reliability_feature_dim
        )

    def retrieval_input_dim(self, space_name: str) -> int:
        numeric_dim = len(self.retrieval_numeric_indices[space_name])
        categorical_dim = sum(
            self.embedding_dimensions[self.categorical_columns[index]]
            for index in self.retrieval_categorical_indices[space_name]
        )
        return numeric_dim + categorical_dim

    def validate(self) -> None:
        """Validate dimensions and notebook architecture invariants."""
        if self.branch_names != BRANCH_NAMES:
            raise ValueError(f'branch_names must be {BRANCH_NAMES!r}.')
        if self.retrieval_space_names != RETRIEVAL_SPACE_NAMES:
            raise ValueError(f'retrieval_space_names must be {RETRIEVAL_SPACE_NAMES!r}.')
        if set(self.category_cardinalities) != set(self.categorical_columns):
            raise ValueError('Category cardinalities must cover categorical columns.')
        if set(self.embedding_dimensions) != set(self.categorical_columns):
            raise ValueError('Embedding dimensions must cover categorical columns.')
        if any(value < 1 for value in self.category_cardinalities.values()):
            raise ValueError('Category cardinalities must include the unknown index.')
        if self.k_search < self.k_candidates or self.k_candidates < 1:
            raise ValueError('Expected k_search >= k_candidates >= 1.')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        if self.log_transform not in {'log', 'log1p'}:
            raise ValueError('log_transform must be log or log1p.')
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError('Split fractions must sum to one.')
        for space_name in self.retrieval_space_names:
            if space_name not in self.retrieval_numeric_indices:
                raise ValueError(f'Missing numeric indices for {space_name}.')
            if space_name not in self.retrieval_categorical_indices:
                raise ValueError(f'Missing categorical indices for {space_name}.')

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> HDBCompNetConfig:
        """Build a config, deriving deterministic column-index mappings if omitted."""
        data = dict(values)
        data['numeric_columns'] = tuple(data['numeric_columns'])
        data['categorical_columns'] = tuple(data['categorical_columns'])
        numeric_indices = data.get('numeric_column_indices') or {
            column: index for index, column in enumerate(data['numeric_columns'])
        }
        categorical_indices = data.get('categorical_column_indices') or {
            column: index for index, column in enumerate(data['categorical_columns'])
        }
        data['numeric_column_indices'] = numeric_indices
        data['categorical_column_indices'] = categorical_indices
        data['retrieval_numeric_indices'] = data.get('retrieval_numeric_indices') or {
            space_name: tuple(
                numeric_indices[column] for column in RETRIEVAL_SPACES[space_name]['numeric']
            )
            for space_name in RETRIEVAL_SPACE_NAMES
        }
        data['retrieval_categorical_indices'] = data.get('retrieval_categorical_indices') or {
            space_name: tuple(
                categorical_indices[column]
                for column in RETRIEVAL_CATEGORICAL_COLUMNS[space_name]
            )
            for space_name in RETRIEVAL_SPACE_NAMES
        }
        known_fields = {field.name for field in fields(cls)}
        unexpected = set(data) - known_fields
        if unexpected:
            raise ValueError(f'Unknown configuration fields: {sorted(unexpected)}')
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe representation."""
        return {
            field.name: convert_to_serializable(getattr(self, field.name))
            for field in fields(self)
        }


def load_yaml_config(path: str | Path) -> HDBCompNetConfig:
    """Load and validate a configuration without performing other I/O."""
    with Path(path).open(encoding='utf-8') as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise TypeError('The YAML document must contain a mapping.')
    return HDBCompNetConfig.from_dict(values)


def save_yaml_config(config: HDBCompNetConfig, path: str | Path) -> None:
    """Save a configuration as portable YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False)
