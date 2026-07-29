# HDB-CompNet stages 1–2 refactoring report

## Source of truth

The selected source is the task attachment
`/Users/pveshmeev/Downloads/HDB_CompNet(1).ipynb`, SHA-256
`dd9d10f9c4c912d680d6fa125ec17debed0c73920cf08025c25b2c85b6c7acf8`.
Its metadata identifies a Python 3 Colab notebook with an L4 GPU. It contains
experiment 2 and later test outputs, and has a later filesystem timestamp than the
other similarly named local notebook. See `docs/notebook_inventory.md`.

## Repository audit

The starting Git repository had no commits, no tracked files, no remote, no
`.gitignore`, and no working files. It was an unborn `master` branch. Work was
moved to the separate unborn branch `refactor/stages-1-2` before repository files
were created. There were no pre-existing uncommitted user changes to preserve.
Seven logical commits were created locally. No push or pull request was created:
`origin` is not configured and the local GitHub CLI credential is invalid.

A 421 MB source CSV existed in Downloads and was used read-only to verify the
notebook-derived column order, split sizes, and train category cardinalities. It
was not copied into the repository.

## Archived artifacts

`notebooks/archive/HDB_CompNet_original.ipynb` is a byte-for-byte copy of the
selected source. Both files have the same SHA-256 hash above and `cmp` returned
success. Outputs and metadata were not cleared or rewritten.

## New package structure

- `config.py`: model, retrieval, cache, split, training, AMP, and dataset defaults;
  validation plus YAML round trips.
- `data/`: explicit downloads, fixed split, fitted preprocessing, categories,
  retrieval matrices, train memory, search, typed caches, datasets, and collation.
- `model/`: shared blocks, global/retrieval encoders, retrieval branch, gating,
  and the complete model.
- `training/`: loss, multi-scale metrics, epoch history, atomic checkpoints, AMP
  train/evaluate, fit, and test evaluation.
- `interpretability/`: branch/reliability collection and fixed-candidate target
  numeric/categorical SHAP.
- `visualization/`: training, gate, retrieval, and SHAP plots.
- `utils/`: paths, serialization, and reproducibility.

The target decomposition was followed. `BaselineRegressionBranch` stays in
`model/encoders.py` because it is small and directly consumes the global
representation; a separate one-class module would not be substantive.

## Configuration extraction

`configs/baseline.yaml` reproduces the first complete configuration: 30 epochs,
learning rate 0.0003, minimum learning rate 0.000001, dropout 0.1, weight decay
0.00001, auxiliary weight 0.1, scheduler patience 2, and early-stop patience 5.

`configs/experiment_2.yaml` applies the executed cell-181 changes: 60 epochs,
learning rate 0.00015, minimum learning rate 0.0000005, dropout 0.075, weight
decay 0.000005, auxiliary weight 0.05, scheduler patience 4, and early-stop
patience 12.

The 56 numeric columns follow the actual post-feature-engineering DataFrame order.
Train-only category cardinalities were independently reproduced with the local
source CSV and exact split: town 27, flat type 8, flat model 22, nearest hospital
type 4, and nearest hospital category 4, each including unknown index 0.

## Data pipeline

The split creates `sale_year` only when absent and reproduces train 603,532,
validation 86,219, and test 172,438 rows with random state 42 and stratification.
Frames are copied and reset as in the notebook.

The dataset already contains `log_resale_price`; sample values and
`LOG_TRANSFORM = 'log'` show that it is `np.log(resale_price)`. The target scaler
fits this column only on train. Inverse transformation first reverses
standardization and then uses `np.exp`. No conflicting `log1p` experiment setting
was found.

Numeric scaling fits only train. Coordinates use one isotropic train scale.
Commercial, market/hawker, miscellaneous, multistorey carpark, precinct pavilion,
month sine, and month cosine pass through unscaled. Reference copies retain resale
ID, month index, floor area, flat type, and scaled target.

`town` is used. `bldg_contract_town` and `onemap_status` are excluded.
One-hot retrieval encoding and integer embedding mappings remain separate, and
both fit train only.

Structural and spatial retrieval matrices remain CSR; market remains contiguous
float32. `ReferenceMemory` contains only train rows and keeps resale IDs, months,
areas, flat-type indices, scaled targets, model numeric/category arrays, and all
three retrieval matrices.

## Cache redesign

The notebook builder returned `InitialNeighborCache` and `FinalCandidateCache`,
whereas later restoration returned nested dictionaries of open memmaps. The saved
cell sequence then passed those dictionaries to a collator expecting `load_*`
methods. Successful later outputs imply extra Colab state not represented by the
linear notebook.

The package uses one format:

- both builders and restorers return `dict[split][space]` containing the same
  frozen path-based dataclasses;
- all `load_*` methods call `np.load(..., mmap_mode='r', allow_pickle=False)`;
- a split collator receives only `dict[space, FinalCandidateCache]`;
- each worker lazily opens and retains its own memmaps.

No `Restored*` type, dynamic nested-dictionary inspection, signature guessing,
late attribute binding, or class monkey-patching exists in core code.

## Model extraction

The package retains shared categorical embeddings; global encoder and baseline;
three independent retrieval branches; independent target and candidate encoders
within each branch; pair interactions; scoring; nonnegative learned temporal
penalty; safe masked softmax; candidate-target aggregation; reliability; and
availability-aware gating.

Each branch prediction is an attention-weighted sum of unadjusted train-memory
`target_scaled` values. The final prediction is exactly the four weighted branch
predictions. There is no price adjustment, post-gating residual, multi-head
attention, transformer, or extra regression head.

The notebook does not explicitly zero-initialize the last gating layer; it uses
PyTorch default initialization. The package preserves that factual behavior.

## Training extraction

The main loss is Huber. Auxiliary Huber losses include baseline and only available
retrieval observations, with the notebook auxiliary weight. AMP, `GradScaler`,
gradient clipping, AdamW, ReduceLROnPlateau, early stopping, history, and resume
state are retained. Checkpoints save state dictionaries atomically. Best selection
uses validation `main_loss`; test evaluation occurs only after the selected best
checkpoint is restored.

Metrics are calculated from individual predictions in scaled, log, and SGD price
space. The package does not derive log errors from aggregated price errors.

## Interpretability extraction

The collector retains branch predictions, gate weights, weighted contributions,
availability, all five reliability values, counts, raw mean distance/time gap, and
temporal decay. GradientSHAP for target numerics and KernelSHAP for target category
indices retain fixed retrieval candidates. Captum imports lazily and SHAP never
runs on import. Attention, gate relationships, and SHAP values are descriptive
model diagnostics, not causal estimates.

## Compatibility notes

Public model output keys and the main train/evaluate function names are retained.
The intended API change is that caches are always dataclass descriptors and a
collator receives one split's cache mapping explicitly. Loader construction also
takes split data and memory explicitly instead of notebook globals.

Core paths are `Path` values or explicit function inputs. No core file imports
`google.colab` or contains `/content/hdb_compnet`.

## Verification

| Command/check | Result | Status | Reason if skipped |
|---|---|---|---|
| `git diff --check` | no whitespace errors | pass | |
| `PYTHONPYCACHEPREFIX=/private/tmp/hdb_pycache python -m compileall -q src` | no syntax errors | pass | system Python cache path was sandboxed, so an allowed cache prefix was used |
| `python -m pip install -e . --no-deps` | editable wheel built and installed | pass | initial sandboxed attempt could not resolve build dependencies; approved network rerun passed |
| required public imports | imported under Python 3.11 | pass | |
| both YAML files load and validate | both valid | pass | |
| model construction from both YAML files | 296,099 parameters each | pass | |
| synthetic forward | finite prediction; no NaN; gates sum to 1; unavailable gates zero | pass | exact batch structure came from the collator/model interfaces |
| cache mmap interface | all five arrays loaded as read-only `np.memmap` | pass | |
| `pytest -q` | 3 passed | pass | |
| `ruff format --check src` | 35 files formatted | pass | |
| `ruff check src tests` | no findings | pass | |
| `python -m build` | sdist and wheel built | pass | setuptools emitted a non-failing future license-table deprecation warning |
| Colab/core path grep | no matches | pass | |
| full nearest-neighbour generation | not run | skipped | full 862,189-row workload intentionally excluded |
| full training/test reproduction | not run | skipped | lengthy GPU/data experiment intentionally excluded |
| SHAP reproduction | not run | skipped | lengthy Captum analysis intentionally excluded |

## Security and large-file audit

| Risk type | Path | Safe action |
|---|---|---|
| large source dataset (421 MB) | external Downloads CSV | read only for schema/split audit; not copied or tracked |
| intentional notebook archive (about 5.1 MB) | `notebooks/archive/HDB_CompNet_original.ipynb` | retained as required; hash verified |
| Kaggle credential references | `.env.example`, docs, downloader, archived notebook | names only; no values added |
| generated caches/checkpoints/data | repository ignore rules | excluded by `.gitignore` |
| OS metadata | `src/.DS_Store` | ignored and untracked |

No tracked file existed at the start. No actual credential value, private key,
checkpoint, NumPy cache, parquet file, or CSV was added.

## Known limitations

- The full pipeline was not retrained or tested against the original checkpoint.
- Historical experiment metrics are preserved in the notebook, but numerical
  equivalence requires a separate full run on the original data and compute.
- The saved notebook's successful final outputs cannot be reconciled linearly with
  its incompatible late cache restoration cells; the package resolves the
  interface structurally rather than recreating hidden Colab state.
- Full retrieval cache generation and Captum analysis were not repeated.
- Publishing is pending a configured remote and renewed GitHub authentication.

## Deferred stages

CLI, a full unit-test suite, CI, Docker, MLflow, and DVC are deliberately not part
of stages 1–2. No API, dashboard, deployment, package publication, new features,
hyperparameter search, or architecture changes were added.

## Suggested next steps

1. Run a stage-3 end-to-end equivalence experiment on the original split and
   compare predictions, branch diagnostics, and metrics.
2. Add focused regression fixtures from a small, legally redistributable sample.
3. Add CI for Python 3.11/3.12 lint, test, and build checks.
4. Add a CLI only after the data/cache artifact contract is finalized.
