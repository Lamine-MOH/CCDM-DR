# Embedding-map metrics — provenance

These two tables are the exact `embedding_metrics.csv` and
`pairwise_metrics.csv` from the **2026-10-27 APTOS-128** run of
`analysis/embedding_map.py`:

- seed 111, 48×48 pixels, PCA-64, UMAP + t-SNE
- 500 images per (source, grade)
- sources: `real`, `real_test`, `aug_train`, `aug_popular`, `blendA`,
  `blendA_g3cs4`, `cs1.5`, `cs4.0`

They were verified **byte-identical** against the rerun in
`experiments_outputs/embedding_map_rerun/`. Interpretation, figures and
caveats are in [`../EMBEDDING_MAP_FINDINGS.md`](../EMBEDDING_MAP_FINDINGS.md).
The feature cache and PNGs remain local-only (`experiments_outputs/` is
gitignored).
