# qaoa-rcol audit package

This CPU package reproduces the negative audit in *A Negative Audit of
Low-Rank Matrix Priors for QAOA Parameter Prediction*.

It provides:

- authenticated Stanford Gset G1 and G81 fetchers;
- exact depth-one, ten-qubit QAOA grid labels;
- constant, ridge, k-nearest-neighbor, and RBF kernel-ridge baselines;
- the transductive rank-stability blend under audit;
- complete 48-record dense operator and normalization audits;
- conditional two-rank and QAOA commutator-bound tests;
- a fair optimized-dense runtime sweep;
- four evidence-only line figures and deterministic JSON replay.

Install and verify from the repository root:

```bash
python -m pip install "./code[test]"
qaoa-rcol-reproduce --fetch-data
qaoa-rcol-reproduce --verify
python -m pytest -q code/tests
python code/scripts/validate_release.py
```

Regenerate the evidence:

```bash
qaoa-rcol-reproduce --write
```

Set `HQML_G1_PATH` and `HQML_G81_PATH`, or use `--data-path` and
`--scaling-data-path`, to choose external cache paths. Inputs are accepted only
when their SHA-256 digests match the manifest.

Important scope: sampled rows/right-hand sides do not certify an operator norm;
the two-rank inequality is conditional on independently checked contraction;
the learner is transductive; and runtime measurements are not locked or
hardware independent. The strongest baseline, kNN, substantially outperforms
the audited blend.
