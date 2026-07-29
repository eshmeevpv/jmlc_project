# HDB-CompNet contributor rules

- Treat `notebooks/archive/HDB_CompNet_original.ipynb` as the historical source and `src/hdb_compnet` as the maintained implementation.
- Do not change the model architecture without a dedicated task.
- Build `ReferenceMemory` from train data only.
- Use `town`; do not use `bldg_contract_town`.
- Version 0.1 has neither comparable-price adjustment nor a final residual network.
- Use `pathlib.Path`, public type hints, and single quotes.
- Do not create one-off variables solely for column names.
- Never commit data, credentials, caches, checkpoints, or experiment artifacts.
- Do not run the full training pipeline in routine checks.
- Report only measured metrics; never substitute estimates.
