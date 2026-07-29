# Notebook-to-package map

The source cell index is zero-based, matching `nbformat` iteration.

| Notebook object or section | Cell / execution | New module | New name | Status | Notes |
|---|---:|---|---|---|---|
| Kaggle source download | 2–3 / 2–3 | `data/download.py` | `download_kaggle_dataset_file` | Moved | Environment/standard Kaggle credentials; no Colab import |
| cache dataset download | 53–57 / 92–96 | `data/download.py` | `download_kaggle_cache_artifacts` | Moved | Explicit destination and force |
| fixed split | 7–10 / 61–64 | `data/splitting.py` | `split_hdb_transactions` | Moved | 70/10/20, `sale_year`, seed 42 |
| month features | 16 / 67 | `data/preprocessing.py` | `add_month_features` | Moved | `month_index`, sine, cosine |
| effective HDB LTV | 18–19 / 60, 69 | `data/preprocessing.py` | `add_hdb_ltv_effective` | Moved | Exact flat-type selection |
| target and numeric scalers | 14, 17, 20–22 / 66–72 | `data/preprocessing.py` | `NumericPreprocessingArtifacts`, `prepare_numeric_splits` | Moved | All fit train-only |
| model category mappings | 25–28 / 76–79 | `data/categorical.py` | `CategoricalArtifacts` | Moved | Unknown index 0 |
| retrieval one-hot | 24 / 75 | `data/categorical.py` | `fit_categorical_artifacts` | Moved | Separate train-only encoders |
| retrieval schema/matrices | 12, 30–31 / 65, 80–82 | `config.py`, `data/retrieval_features.py` | `RETRIEVAL_SPACES`, `build_retrieval_matrices` | Moved | CSR structural/spatial; dense market |
| train memory | 33–34 / 83–84 | `data/reference_memory.py` | `ReferenceMemory`, `build_reference_memory` | Moved | Global indices are train-memory positions |
| grouped indices | 36–38 / 85–87 | `data/neighbor_search.py` | `NeighborGroupIndex`, `build_retrieval_indices` | Moved | Per flat type and space |
| initial cache | 41–43 / 125–126, 46 | `data/cache.py`, `data/neighbor_search.py` | `InitialNeighborCache`, `build_initial_neighbor_caches` | Moved | Typed path interface |
| final filtering/cache | 45–47 / 127–128, 50 | `data/cache.py`, `data/neighbor_search.py` | `FinalCandidateCache`, `build_final_candidate_caches` | Moved | Strict earlier month and area backoff |
| restored cache dictionaries | 59–61 / 133–135 | `data/cache.py` | `restore_*_caches` | Replaced | Restorers now return the builder types |
| model split container | 63–65 / 129–130, 136 | `data/datasets.py` | `ModelSplitData`, `build_model_split_data` | Moved | Cache mapping passed to collator explicitly |
| index dataset | 66–67 / 137–138 | `data/datasets.py` | `TransactionIndexDataset` | Moved | Returns one integer |
| candidate collator | 68–69 / 139–140 | `data/collate.py` | `CandidateCacheArrays`, `HDBCompNetCollator` | Moved | Lazy worker-local memmaps |
| loader factory | 112 / 162 | `data/collate.py` | `create_hdb_compnet_dataloaders` | Moved | Globals replaced by explicit inputs |
| configuration | 72–74 / 141–143 | `config.py` | `HDBCompNetConfig` | Moved | Added validation and YAML serialization |
| categorical embeddings | 76 / 144 | `model/blocks.py` | `CategoricalEmbeddingBlock` | Moved | Shared tables |
| MLP | 78 / 145 | `model/blocks.py` | `MLPBlock` | Moved | Numerically unchanged order |
| global encoder | 80 / 146 | `model/encoders.py` | `GlobalPropertyEncoder` | Moved | Same dimensions |
| baseline branch | 82 / 147 | `model/encoders.py` | `BaselineRegressionBranch` | Moved | Same regression head |
| retrieval adapters/preparer | 85–86 / 148–149 | `model/encoders.py` | same | Moved | Same numeric/category selection |
| retrieval encoders | 88 / 150 | `model/encoders.py` | `RetrievalRepresentationEncoders` | Moved | Independent target/candidate weights |
| pair features and scoring | 90–92 / 151–152 | `model/retrieval.py` | `PairFeatureBuilder`, `PairwiseScoreNetwork` | Moved | Same interactions |
| temporal penalty | 94 / 153 | `model/retrieval.py` | `TemporalRecencyPenalty` | Moved | Softplus nonnegative rate |
| masked softmax | 96 / 154 | `model/blocks.py` | `safe_masked_softmax` | Moved | All-masked row returns zero |
| retrieval aggregation/reliability | 98–100 / 155–156 | `model/retrieval.py` | same | Moved | Candidate target weighted sum |
| full retrieval branch | 102 / 157 | `model/retrieval.py` | `RetrievalBranch` | Moved | Same output keys |
| gating | 104–106 / 158–159 | `model/gating.py` | `GatingNetwork`, `combine_branch_predictions` | Moved | Default final-layer initialization |
| full network | 108 / 160 | `model/hdb_compnet.py` | `HDBCompNet` | Moved | Same output dictionary |
| objective | 110 / 161 | `training/losses.py` | `calculate_hdb_compnet_loss` | Moved | Masked auxiliary Huber |
| metrics | 117–118 / 164–165 | `training/metrics.py` | same | Moved | Scaled, log, price |
| epoch aggregation/history | 120, 126 / 166, 169 | `training/history.py` | same | Moved | State dict retained |
| train/evaluate loops | 115, 122, 124 / 163, 167–168 | `training/engine.py` | same | Moved | AMP and clipping retained |
| checkpoint/early stop | 130–131 / 171–172 | `training/checkpointing.py` | same | Moved | Atomic state-dict saves |
| fit and test | 133, 136 / 173–174 | `training/engine.py` | same | Moved | Best validation main-loss checkpoint |
| training plots | 128 / 170 | `visualization/training.py` | same | Moved | No notebook globals |
| interpretation collector | 157 / 193 | `interpretability/gating_analysis.py` | same | Moved | Predictions, gates, contributions, reliability |
| gate/retrieval analysis | 138–142 / 175–179 | `interpretability/gating_analysis.py`, `visualization/interpretation.py` | public helper names | Moved | Relationships are descriptive, not causal |
| SHAP batch helpers/wrappers | 146–151 / 182–187 | `interpretability/shap_analysis.py` | same | Moved | Candidates remain fixed |
| numeric/categorical SHAP | 152–154, 156 / 188–190, 192 | `interpretability/shap_analysis.py` | same | Moved | Captum imported lazily |
| SHAP plot | 155 / 191 | `visualization/interpretation.py` | `plot_target_shap_importance` | Moved | No global state |
| seed helper | 162 / 199 | `utils/reproducibility.py` | `set_experiment_seed` | Moved | Determinism parameterized |
| experiment 1 | 161–177 / 198–256 | `configs/baseline.yaml` | baseline config | Extracted | Historical run remains in archive |
| experiment 2 | 180–196 / 262–294 | `configs/experiment_2.yaml` | experiment-2 config | Extracted | Historical run remains in archive |

No significant working implementation is intentionally left only in the notebook.
The interactive experiment-launch cells remain historical rather than becoming a
CLI, which is explicitly deferred.
