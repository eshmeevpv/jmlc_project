# HDB-CompNet notebook inventory

## Selected source

The selected source is `/Users/pveshmeev/Downloads/HDB_CompNet(1).ipynb`,
SHA-256 `dd9d10f9c4c912d680d6fa125ec17debed0c73920cf08025c25b2c85b6c7acf8`.
It is the notebook attached to the task, has the later filesystem timestamp, and
contains both complete experiments and their test outputs. A different, older
`HDB_CompNet.ipynb` was present in Downloads and has hash
`4180687cf22fbc82e42bdf6dd7bb14253986c36edfb3b2c5560dcca34498fdf8`.

The selected notebook was parsed with `nbformat` and every Python code cell was
parsed with `ast`. It contains 198 cells: 145 code cells and 53 Markdown cells.
The kernel metadata is Python 3 with a Colab L4 GPU. Cells 1 and 144 are IPython
`!pip` magics and therefore expected AST syntax exceptions.

## Section and execution overview

| Notebook cells | Execution counts | Purpose |
|---|---:|---|
| 0–4 | 1–4 | Colab/Kaggle download and CSV load |
| 5–10 | 61–64 | 70/10/20 split by `sale_year`, random state 42 |
| 11–38 | 60, 65–87 | retrieval schema, target/numeric/category preparation, train memory and indices |
| 39–61 | 46, 50–53, 92–96, 125–135 | initial/final candidate caches and external cache restoration |
| 62–69 | 129–140 | split containers, index dataset, collator |
| 70–110 | 141–161 | configuration, four-branch model, and loss |
| 111–136 | 162–174 | loaders, metrics, loops, history, plots, checkpointing, fit/test |
| 137–157 | 175–193 | gating/retrieval interpretation and fixed-candidate SHAP |
| 158–177 | 198–256 and 238–253 | full experiment 1 and analysis |
| 178–196 | 261–294 | full experiment 2 and analysis |

Execution order is not cell order. The inventory therefore treats the executed
experiment-2 configuration in cell 181 (execution 263) and the model/training
definitions used by cells 183–188 as authoritative.

## Function definitions

| Cell (execution) | Functions |
|---|---|
| 18 (60) | `add_hdb_ltv_effective` |
| 21 (71) | `transform_numeric_features` |
| 27 (78) | `encode_categories` |
| 30 (80) | `build_retrieval_matrices` |
| 37 (86) | `build_retrieval_indices` |
| 42 (126) | `build_initial_neighbor_caches` |
| 46 (128) | `build_final_candidate_caches` |
| 55 (94) | `find_cache_source` |
| 59 (133) | `load_cache_arrays` |
| 64 (130) | `build_model_split_data` |
| 96 (154) | `safe_masked_softmax` |
| 106 (159) | `combine_branch_predictions` |
| 110 (161) | `calculate_hdb_compnet_loss` |
| 112 (162) | `create_hdb_compnet_dataloaders` |
| 115 (163) | `move_batch_to_device` |
| 117 (164) | `to_numpy_1d`, `inverse_transform_target` |
| 118 (165) | `calculate_regression_metrics` |
| 122 (167) | `train_one_epoch` |
| 124 (168) | `evaluate_one_epoch` |
| 128 (170) | `get_training_history_data`, `plot_history_series`, `plot_model_losses`, `plot_branch_losses`, `plot_branch_availability`, `plot_validation_metrics`, `plot_learning_rate`, `plot_training_progress`, `plot_validation_test_metric_comparison`, `plot_final_validation_test_comparison` |
| 130 (171) | `convert_to_serializable`, `config_to_dict`, `compact_epoch_statistics` |
| 133 (173) | `fit_hdb_compnet` |
| 136 (174) | `evaluate_hdb_compnet_on_test` |
| 138 (175) | `_get_gate_columns`, `_get_retrieval_analysis_values` |
| 139 (176) | `_calculate_binned_relationship` |
| 140 (177) | `_plot_gate_relationship` |
| 141 (178) | `plot_gate_weight_analysis` |
| 142 (179) | `plot_retrieval_branch_analysis` |
| 146 (182) | `slice_nested_batch` |
| 147 (183) | `expand_single_example_batch` |
| 148 (184) | `iterate_single_example_batches` |
| 151 (187) | `create_target_shap_baselines` |
| 152 (188) | `collect_target_numeric_shap` |
| 153 (189) | `collect_target_categorical_shap` |
| 154 (190) | `combine_target_shap_importance` |
| 155 (191) | `plot_target_shap_importance` |
| 156 (192) | `calculate_target_shap_analysis` |
| 157 (193) | `collect_hdb_compnet_interpretation_outputs` |
| 162 (199) | `set_experiment_seed` |

## Class definitions

| Cell (execution) | Classes |
|---|---|
| 33 (83) | `ReferenceMemory` |
| 36 (85) | `NeighborGroupIndex` |
| 41 (125) | `InitialNeighborCache` |
| 45 (127) | `FinalCandidateCache` |
| 63 (129) | `ModelSplitData`, `CandidateCacheArrays` |
| 66 (137) | `TransactionIndexDataset` |
| 68 (139) | `HDBCompNetCollator` |
| 72 (141) | `HDBCompNetConfig` |
| 76 (144) | `CategoricalEmbeddingBlock` |
| 78 (145) | `MLPBlock` |
| 80 (146) | `GlobalPropertyEncoder` |
| 82 (147) | `BaselineRegressionBranch` |
| 85 (148) | `RetrievalInputAdapter` |
| 86 (149) | `RetrievalInputPreparer` |
| 88 (150) | `RetrievalRepresentationEncoders` |
| 90 (151) | `PairFeatureBuilder` |
| 92 (152) | `PairwiseScoreNetwork` |
| 94 (153) | `TemporalRecencyPenalty` |
| 98 (155) | `RetrievalPredictionAggregator` |
| 100 (156) | `RetrievalReliabilityEstimator` |
| 102 (157) | `RetrievalBranch` |
| 104 (158) | `GatingNetwork` |
| 108 (160) | `HDBCompNet` |
| 120 (166) | `EpochStatisticsAggregator` |
| 126 (169) | `HDBCompNetTrainingHistory` |
| 131 (172) | `HDBCompNetCheckpointManager` |
| 149 (185) | `TargetNumericSHAPWrapper` |
| 150 (186) | `TargetCategoricalSHAPWrapper` |

## Imports and direct dependencies

AST found imports from NumPy, pandas, SciPy, scikit-learn, PyTorch, Matplotlib,
Captum, pathlib, dataclasses, and standard-library modules. `google.colab` and
IPython are notebook-adapter concerns and are not core package dependencies.
Kaggle is an explicit optional downloader dependency. Captum is also an extra
because it is imported only when SHAP analysis is requested. Joblib and PyArrow
are not declared because the extracted package does not import them directly.

## Repeated definitions and state

No function or class name is defined more than once. Repeated global assignments
include `df_train`, `df_val`, `df_test`, `initial_neighbor_caches`,
`final_candidate_caches`, `CACHE_DIR`, `RESULTS_DIR`, and the two experiment model
variables. The later assignments are experiment/runtime state, not alternative
implementations.

The cache state is contradictory:

- cells 41–47 define and build typed cache objects with `load_*()` methods;
- cells 59–61 later restore nested dictionaries of already-open memmaps;
- cells 64–69 later construct split/collator objects whose collator expects
  `load_*()` methods.

The attached notebook contains no AST-visible monkey-patch that reconciles those
interfaces, although later successful experiment outputs exist. This indicates
Colab state not fully represented by the saved linear cell sequence. The package
selects the stable typed interface already established by cells 41 and 45 and
makes both builders and restorers return it.

## Working implementation choices

- The target scaler fits only `df_train[['log_resale_price']]` in cell 14.
- `LOG_TRANSFORM = 'log'` in cell 161, and sample values equal
  `np.log(resale_price)`, not `np.log1p`.
- The seven flat-type HDB LTV columns are replaced by `hdb_ltv_effective`.
- `town` is in `CATEGORY_COLUMNS`; `bldg_contract_town` and `onemap_status` are
  not used.
- One-hot encoders are fitted per retrieval space on train; integer mappings with
  unknown index 0 are fitted separately for model embeddings.
- `ReferenceMemory` is constructed only from train arrays.
- Initial retrieval is brute-force Euclidean within `flat_type`, up to 128.
  Self and nonhistorical candidates are removed only afterward; 15/20/25% area
  backoff produces up to 32 candidates.
- Target and candidate encoders have independent parameters. Every candidate in a
  branch passes through the same candidate encoder.
- The final gating linear layer uses PyTorch's default initialization. There is no
  explicit zero initialization in the notebook.
- Final output is only the four-way gated sum; there is no residual correction.
- Experiment 2 is the baseline config with epochs 60, learning rate 0.00015,
  minimum learning rate 0.0000005, weight decay 0.000005, dropout 0.075,
  auxiliary weight 0.05, scheduler patience 4, and early-stop patience 12.

## Notebook-to-module sections

| Notebook section | Package area |
|---|---|
| download and split | `hdb_compnet.data.download`, `data.splitting` |
| target/numeric construction | `data.preprocessing` |
| category indices and one-hot retrieval | `data.categorical` |
| retrieval matrices and memory | `data.retrieval_features`, `data.reference_memory` |
| neighbour search and caches | `data.neighbor_search`, `data.cache` |
| datasets, collator, loaders | `data.datasets`, `data.collate` |
| configuration | `hdb_compnet.config` and `configs/*.yaml` |
| neural network | `model.blocks`, `model.encoders`, `model.retrieval`, `model.gating`, `model.hdb_compnet` |
| loss, metrics, loops, checkpoints | `training.*` |
| interpretation and SHAP | `interpretability.*` |
| plots | `visualization.*` |
| seed/path/serialization helpers | `utils.*` |
