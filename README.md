# Tie-Complete Nearest-Signature Transfer for Depth-One QAOA

This repository contains a deterministic study of parameter transfer when graph descriptors create exact nearest-neighbor ties. The proposed rule averages every training label at the selected neighbor radius, making prediction invariant to training-row order.

The evidence uses 48 ten-vertex neighborhoods from checksum-authenticated Stanford Gset G1 and exact 81-point depth-one QAOA grids. Row-held-out folds give mean angle MSE 0.00072 for tie completion and 0.00137 for ordinary nearest neighbors. Under the primary signature-grouped protocol, training and test signatures are disjoint; both methods have MSE 0.00780 and relative regret 0.00267. The equality is retained because it shows that tie completion improves duplicate handling, not unseen-signature extrapolation.

## Reproduction

    python -m venv .venv
    .venv/bin/pip install -e 'code[test]'
    .venv/bin/qaoa-rcol-reproduce --verify
    .venv/bin/pytest code/tests
    tectonic main.tex

The replay authenticates G1 and G81, reconstructs all graph neighborhoods and QAOA references, regenerates the fold diagnostics and timing plot, and verifies all non-timing fields against the locked result.
