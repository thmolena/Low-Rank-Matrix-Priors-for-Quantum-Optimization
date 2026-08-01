# Data provenance

**Dataset:** Stanford Gset G1 source graph and G81 application-audit graph.

**Original source:** https://web.stanford.edu/~yyye/yyye/Gset/

**Derived protocol:** Forty-eight ten-vertex G1 neighborhoods, exact depth-one
QAOA labels on a fixed 81-point grid, and four held-out record folds. These are
derived computed tasks, not experimental quantum measurements. G81 supplies
degree/index features for a constructed kernel and no learning labels.

**Integrity:** The evidence manifest pins the official URLs and SHA-256 digests.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
