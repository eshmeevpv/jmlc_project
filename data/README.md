# Data

Large source data and generated artifacts are not stored in Git. The notebook's
source dataset is `paveleshmeev/sgp-resale-prices-90` on Kaggle and the expected
file is `df_common_sample_complete_case.csv`. The optional retrieval-cache dataset
is `paveleshmeev/hdb-compnet-cache`.

The downloader reads standard Kaggle configuration or these environment variables:

```text
KAGGLE_USERNAME
KAGGLE_KEY
```

Never place their values in this repository. By convention, downloaded source
files go to `data/raw/`, fitted or transformed tables to `data/processed/`, and
temporary intermediate data to `data/interim/`. Retrieval arrays go under
`cache/`. These paths are ignored by Git.

`ReferenceMemory`, initial-neighbour caches, and final-candidate caches are built
locally from the fixed split. They may instead be restored from external storage,
but restoration produces the same typed cache interface as local construction.
