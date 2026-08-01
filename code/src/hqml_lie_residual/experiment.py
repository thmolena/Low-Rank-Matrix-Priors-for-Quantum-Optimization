from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .bounds import qaoa_depth_one_lipschitz_constant, qaoa_regret_bound
from .matrix_free import (
    dense_hierarchy_audit,
    dense_reference_prior,
    fair_dense_runtime_audit,
    hierarchical_prior_bundle,
    matrix_free_certificate,
    select_rank_counterfactual_blend,
)

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "results").is_dir() else SOURCE_ROOT
GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/G1"
GSET_SHA256 = "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665"
GSET_G81_URL = "https://web.stanford.edu/~yyye/yyye/Gset/G81"
GSET_G81_SHA256 = "74e69d2f5228774cedbdb86da14debf08023556f1d7693b7346ca13df7594d5a"
CONFIG = {
    "key": "lie_residual",
    "main": "Low-Rank-Matrix-Priors-for-Quantum-Optimization/main.tex",
    "title": "Low-Rank Matrix Priors for Quantum Approximate Optimization Parameters",
    "domain": "qaoa",
    "task": "matrix-free QAOA parameter-prior recovery",
    "question": (
        "Does a transductive hierarchical kernel smoother remain competitive "
        "with ordinary inductive baselines on G1-derived depth-one QAOA tasks?"
    ),
    "mechanism": (
        "A butterfly feature map orders task signatures; entry-oracle cross "
        "approximation builds low-rank off-diagonal blocks without a global "
        "dense matrix."
    ),
    "loss": (
        "development loss plus rank-counterfactual stability, with a sampled "
        "operator diagnostic, storage gate, and capped prior weight"
    ),
    "seed": 1101,
    "package": "hqml_lie_residual",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(filename: str, environment: str, explicit: Path | None) -> Path:
    """Resolve an external data cache without writing into the package."""

    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.environ.get(environment)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "hqml_lie_residual" / filename


def _fetch_gset(
    *,
    name: str,
    url: str,
    expected_sha256: str,
    environment: str,
    explicit: Path | None,
) -> Path:
    destination = _cache_path(name, environment, explicit)
    if destination.is_file():
        if sha256(destination) != expected_sha256:
            raise ValueError(
                f"cached {name} SHA-256 mismatch at {destination}; remove the "
                "file before retrying"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    request = Request(
        url,
        headers={"User-Agent": "hqml-lie-residual/1.0 data fetcher"},
    )
    try:
        # The URL is a fixed, HTTPS-only Stanford endpoint rather than user input.
        with urlopen(request, timeout=60) as response, partial.open(  # nosec B310
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        if sha256(partial) != expected_sha256:
            raise ValueError(f"downloaded {name} SHA-256 mismatch")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def fetch_gset_g1(explicit: Path | None = None) -> Path:
    """Fetch and authenticate Stanford Gset G1 when it is not cached."""

    return _fetch_gset(
        name="G1",
        url=GSET_URL,
        expected_sha256=GSET_SHA256,
        environment="HQML_G1_PATH",
        explicit=explicit,
    )


def fetch_gset_g81(explicit: Path | None = None) -> Path:
    """Fetch and authenticate Stanford Gset G81 when it is not cached."""

    return _fetch_gset(
        name="G81",
        url=GSET_G81_URL,
        expected_sha256=GSET_G81_SHA256,
        environment="HQML_G81_PATH",
        explicit=explicit,
    )


def parse_gset(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    with path.open() as f:
        n, m = map(int, f.readline().split()[:2])
        edges = []
        for line in f:
            u, v, w = line.split()
            edges.append((int(u) - 1, int(v) - 1, float(w)))
    if len(edges) != m:
        raise ValueError(f"Gset header declares {m} edges but parsed {len(edges)}")
    if any(u < 0 or v < 0 or u >= n or v >= n or u == v for u, v, _ in edges):
        raise ValueError("Gset contains an invalid endpoint or self-loop")
    return n, edges


def bfs_subgraph(adj: list[list[int]], start: int, size: int) -> list[int]:
    seen = {start}
    queue = [start]
    for u in queue:
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
                if len(queue) == size:
                    return queue
    for v in range(len(adj)):
        if v not in seen:
            queue.append(v)
            if len(queue) == size:
                break
    return queue


def qaoa_energy(edges: list[tuple[int, int]], gamma: float, beta: float, n: int) -> float:
    states = np.arange(1 << n, dtype=np.uint32)
    cost = np.zeros(len(states), dtype=float)
    for u, v in edges:
        cost += ((states >> u) ^ (states >> v)) & 1
    psi = np.exp(-1j * gamma * cost) / math.sqrt(len(states))
    c, s = math.cos(beta), -1j * math.sin(beta)
    for q in range(n):
        block = 1 << q
        for start in range(0, len(states), 2 * block):
            a = psi[start : start + block].copy()
            b = psi[start + block : start + 2 * block].copy()
            psi[start : start + block] = c * a + s * b
            psi[start + block : start + 2 * block] = s * a + c * b
    return float(np.dot(np.abs(psi) ** 2, cost))


def qaoa_moments(
    edges: list[tuple[int, int]], gamma: float, beta: float, n: int
) -> tuple[float, float]:
    states = np.arange(1 << n, dtype=np.uint32)
    cost = np.zeros(len(states), dtype=float)
    for u, v in edges:
        cost += ((states >> u) ^ (states >> v)) & 1
    psi = np.exp(-1j * gamma * cost) / math.sqrt(len(states))
    c, s = math.cos(beta), -1j * math.sin(beta)
    for q in range(n):
        block = 1 << q
        for start in range(0, len(states), 2 * block):
            a = psi[start : start + block].copy()
            b = psi[start + block : start + 2 * block].copy()
            psi[start : start + block] = c * a + s * b
            psi[start + block : start + 2 * block] = s * a + c * b
    probability = np.abs(psi) ** 2
    mean = float(np.dot(probability, cost))
    variance = float(np.dot(probability, (cost - mean) ** 2))
    return mean, variance


def graph_features(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for u, v in edges:
        a[u, v] = a[v, u] = 1.0
    degrees = a.sum(axis=1)
    lap = np.diag(degrees) - a
    eig = np.linalg.eigvalsh(lap)
    triangles = np.trace(a @ a @ a) / 6.0
    return np.array(
        [
            1.0,
            len(edges) / max(n * (n - 1) / 2, 1),
            degrees.mean(),
            degrees.std(),
            degrees.max(),
            triangles,
            eig[1] if len(eig) > 1 else 0.0,
            eig[-1],
        ]
    )


def _clip_angles(prediction: np.ndarray) -> np.ndarray:
    output = np.asarray(prediction, dtype=float).copy()
    output[:, 0] = np.mod(output[:, 0], math.pi)
    output[:, 1] = np.clip(output[:, 1], 0.0, math.pi / 2)
    return output


def _qaoa_regret(
    prediction: np.ndarray,
    test: np.ndarray,
    optima: np.ndarray,
    edge_sets: list[list[tuple[int, int]]],
) -> float:
    regrets = []
    for row, index in enumerate(test):
        achieved = qaoa_energy(
            edge_sets[int(index)],
            float(prediction[row, 0]),
            float(prediction[row, 1]),
            10,
        )
        regrets.append(
            max(
                0.0,
                (float(optima[index]) - achieved)
                / max(float(optima[index]), 1e-12),
            )
        )
    return float(np.mean(regrets))


def _ridge_prediction(
    features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    development: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    best: tuple[float, float] | None = None
    for regularization in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
        matrix = (
            design[fit].T @ design[fit]
            + regularization * penalty
            + 1e-12 * np.eye(design.shape[1])
        )
        coefficient = np.linalg.solve(matrix, design[fit].T @ targets[fit])
        loss = float(
            np.mean((design[development] @ coefficient - targets[development]) ** 2)
        )
        candidate = (loss, float(regularization))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("ridge grid is empty")
    regularization = best[1]
    matrix = (
        design[train].T @ design[train]
        + regularization * penalty
        + 1e-12 * np.eye(design.shape[1])
    )
    coefficient = np.linalg.solve(matrix, design[train].T @ targets[train])
    return (
        design[development] @ np.linalg.solve(
            design[fit].T @ design[fit]
            + regularization * penalty
            + 1e-12 * np.eye(design.shape[1]),
            design[fit].T @ targets[fit],
        ),
        design[test] @ coefficient,
        regularization,
    )


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def _knn_prediction(
    features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    development: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tune an ordinary Euclidean k-nearest-neighbor regressor."""

    candidates = tuple(k for k in (1, 3, 5, 9, 15) if k <= len(fit))
    if not candidates:
        raise ValueError("not enough fitting records for kNN")
    development_order = np.argsort(
        _squared_distances(features[development], features[fit]),
        axis=1,
        kind="mergesort",
    )
    best: tuple[float, int] | None = None
    for k in candidates:
        prediction = targets[fit][development_order[:, :k]].mean(axis=1)
        candidate = (
            float(np.mean((prediction - targets[development]) ** 2)),
            int(k),
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("kNN candidate grid is empty")
    k = best[1]
    test_order = np.argsort(
        _squared_distances(features[test], features[train]),
        axis=1,
        kind="mergesort",
    )
    development_prediction = targets[fit][development_order[:, :k]].mean(axis=1)
    test_prediction = targets[train][test_order[:, :k]].mean(axis=1)
    return development_prediction, test_prediction, k


def _tie_complete_prediction(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    reference_targets: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average every neighbor tied at the k-th distance.

    This removes dependence on the row order when repeated graph descriptors
    occur.  The returned counts expose the effective, query-dependent sample
    size created by a distance tie.
    """

    distances = _squared_distances(query_features, reference_features)
    order = np.argsort(distances, axis=1, kind="mergesort")
    predictions = []
    counts = []
    for row, ranking in zip(distances, order):
        radius = float(row[ranking[k - 1]])
        tolerance = 1e-12 * max(1.0, radius)
        included = row <= radius + tolerance
        predictions.append(reference_targets[included].mean(axis=0))
        counts.append(int(np.sum(included)))
    return np.asarray(predictions), np.asarray(counts, dtype=int)


def _tie_complete_knn_prediction(
    features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    development: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Tune a permutation-invariant, tie-complete neighbor regressor."""

    candidates = tuple(k for k in (1, 3, 5, 9, 15) if k <= len(fit))
    best: tuple[float, int, np.ndarray] | None = None
    for k in candidates:
        prediction, _ = _tie_complete_prediction(
            features[development], features[fit], targets[fit], k
        )
        candidate = (float(np.mean((prediction - targets[development]) ** 2)), int(k), prediction)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("tie-complete neighbor candidate grid is empty")
    k = best[1]
    development_prediction = best[2]
    test_prediction, counts = _tie_complete_prediction(
        features[test], features[train], targets[train], k
    )
    return development_prediction, test_prediction, k, float(np.mean(counts))


def _kernel_ridge_prediction(
    features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    development: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Tune a dense RBF kernel-ridge baseline on development labels."""

    dimension = max(features.shape[1], 1)

    def kernel(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
        return np.exp(
            -_squared_distances(left, right)
            / (2.0 * bandwidth * bandwidth * dimension)
        )

    best: tuple[float, float, float] | None = None
    for bandwidth in (0.25, 0.5, 1.0, 2.0, 4.0):
        fit_kernel = kernel(features[fit], features[fit], bandwidth)
        development_kernel = kernel(
            features[development], features[fit], bandwidth
        )
        for regularization in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0):
            coefficient = np.linalg.solve(
                fit_kernel + regularization * np.eye(len(fit)), targets[fit]
            )
            prediction = development_kernel @ coefficient
            candidate = (
                float(np.mean((prediction - targets[development]) ** 2)),
                float(bandwidth),
                float(regularization),
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("kernel-ridge candidate grid is empty")
    _, bandwidth, regularization = best
    fit_coefficient = np.linalg.solve(
        kernel(features[fit], features[fit], bandwidth)
        + regularization * np.eye(len(fit)),
        targets[fit],
    )
    train_coefficient = np.linalg.solve(
        kernel(features[train], features[train], bandwidth)
        + regularization * np.eye(len(train)),
        targets[train],
    )
    return (
        kernel(features[development], features[fit], bandwidth) @ fit_coefficient,
        kernel(features[test], features[train], bandwidth) @ train_coefficient,
        bandwidth,
        regularization,
    )


def _qaoa_regret_rows(
    prediction: np.ndarray,
    test: np.ndarray,
    optima: np.ndarray,
    edge_sets: list[list[tuple[int, int]]],
) -> np.ndarray:
    values = []
    for row, index in enumerate(test):
        achieved = qaoa_energy(
            edge_sets[int(index)],
            float(prediction[row, 0]),
            float(prediction[row, 1]),
            10,
        )
        values.append(
            max(
                0.0,
                (float(optima[index]) - achieved)
                / max(float(optima[index]), 1e-12),
            )
        )
    return np.asarray(values, dtype=float)


def _qaoa_angle_regret_bounds(
    prediction: np.ndarray,
    test: np.ndarray,
    targets: np.ndarray,
    optima: np.ndarray,
    edge_sets: list[list[tuple[int, int]]],
) -> np.ndarray:
    """Return the elementary commutator/Lipschitz regret upper bounds."""

    bounds = []
    for row, index in enumerate(test):
        edge_count = len(edge_sets[int(index)])
        bounds.append(
            qaoa_regret_bound(
                prediction[row],
                targets[index],
                float(optima[index]),
                float(edge_count),
                10,
            )
        )
    return np.asarray(bounds, dtype=float)


def _evaluate_qaoa_fold(
    features: np.ndarray,
    targets: np.ndarray,
    optima: np.ndarray,
    edge_sets: list[list[tuple[int, int]]],
    fold: int,
) -> dict:
    indices = np.arange(len(features))
    test = indices[indices % 4 == fold]
    train = indices[indices % 4 != fold]
    mean, std = features[train].mean(axis=0), features[train].std(axis=0)
    std[std < 1e-9] = 1.0
    standardized = (features - mean) / std
    development = train[np.arange(len(train)) % 5 == fold % 5]
    fit = np.setdiff1d(train, development)
    fit_mask = np.zeros(len(features), dtype=bool)
    fit_mask[fit] = True
    train_mask = np.zeros(len(features), dtype=bool)
    train_mask[train] = True

    prior = hierarchical_prior_bundle(
        standardized,
        [fit_mask, train_mask],
        targets,
        CONFIG["seed"] + fold,
        bandwidth=0.10,
        leaf_size=8,
        low_rank=2,
        high_rank=4,
    )
    (
        ridge_development,
        ridge_prediction,
        ridge_regularization,
    ) = _ridge_prediction(
        standardized, targets, fit, development, train, test
    )
    knn_development, knn_prediction, knn_neighbors = _knn_prediction(
        standardized, targets, fit, development, train, test
    )
    (
        tie_knn_development,
        tie_knn_prediction,
        tie_knn_neighbors,
        tie_knn_effective_neighbors,
    ) = _tie_complete_knn_prediction(
        standardized, targets, fit, development, train, test
    )
    (
        kernel_ridge_development,
        kernel_ridge_prediction,
        kernel_ridge_bandwidth,
        kernel_ridge_regularization,
    ) = _kernel_ridge_prediction(
        standardized, targets, fit, development, train, test
    )
    selected = select_rank_counterfactual_blend(
        targets[development],
        ridge_development,
        prior["low"][0][development],
        prior["high"][0][development],
        loss="mse",
        sampled_operator_error=prior["metrics"][
            "hmat_sampled_row_relative_error"
        ],
        storage_fraction=prior["metrics"]["hmat_storage_fraction"],
        maximum_alpha=0.20,
    )
    baseline = np.tile(targets[train].mean(axis=0), (len(test), 1))
    alpha = float(selected["alpha"])
    prediction = _clip_angles(
        (1.0 - alpha) * ridge_prediction
        + alpha * prior["high"][1][test]
    )
    prediction_low = _clip_angles(
        (1.0 - alpha) * ridge_prediction
        + alpha * prior["low"][1][test]
    )
    dense_prior = dense_reference_prior(
        standardized, train_mask, targets, bandwidth=0.10
    )
    hierarchy_audit = dense_hierarchy_audit(
        standardized,
        train_mask,
        targets,
        CONFIG["seed"] + fold,
        bandwidth=0.10,
        leaf_size=8,
        low_rank=2,
        high_rank=4,
    )
    dense_prediction = _clip_angles(
        (1.0 - alpha) * ridge_prediction + alpha * dense_prior[test]
    )
    ridge_prediction = _clip_angles(ridge_prediction)
    knn_prediction = _clip_angles(knn_prediction)
    tie_knn_prediction = _clip_angles(tie_knn_prediction)
    kernel_ridge_prediction = _clip_angles(kernel_ridge_prediction)

    baseline_loss = float(np.mean((baseline - targets[test]) ** 2))
    learned_loss = float(np.mean((prediction - targets[test]) ** 2))
    ridge_loss = float(np.mean((ridge_prediction - targets[test]) ** 2))
    knn_loss = float(np.mean((knn_prediction - targets[test]) ** 2))
    tie_knn_loss = float(np.mean((tie_knn_prediction - targets[test]) ** 2))
    tie_knn_regret_rows = _qaoa_regret_rows(
        tie_knn_prediction, test, optima, edge_sets
    )
    tie_knn_bounds = _qaoa_angle_regret_bounds(
        tie_knn_prediction, test, targets, optima, edge_sets
    )
    kernel_ridge_loss = float(
        np.mean((kernel_ridge_prediction - targets[test]) ** 2)
    )
    dense_reference_loss = float(
        np.mean((dense_prediction - targets[test]) ** 2)
    )
    target_scale = max(
        float(np.mean((targets[test] - targets[train].mean(axis=0)) ** 2)),
        1e-8,
    )
    rank_counterfactual_term = float(
        np.mean((prediction - prediction_low) ** 2) / target_scale
    )
    rcol_test_loss = (
        learned_loss
        + float(selected["consistency_weight"]) * rank_counterfactual_term
        + float(selected["certificate_weight"])
        * float(selected["certificate_violation"])
        + float(selected["budget_weight"])
        * float(selected["budget_violation"])
    )
    dense_prediction_error = float(
        np.linalg.norm(prediction - dense_prediction)
        / max(float(np.linalg.norm(dense_prediction)), 1e-15)
    )
    learned_regret_rows = _qaoa_regret_rows(
        prediction, test, optima, edge_sets
    )
    dense_regret_rows = _qaoa_regret_rows(
        dense_prediction, test, optima, edge_sets
    )
    regret_bounds = _qaoa_angle_regret_bounds(
        prediction, test, targets, optima, edge_sets
    )
    chain_coefficients = []
    for index in test:
        edge_count = len(edge_sets[int(index)])
        lipschitz = qaoa_depth_one_lipschitz_constant(float(edge_count), 10)
        chain_coefficients.append(
            lipschitz / max(float(optima[int(index)]), 1e-12)
        )
    row_bounds = np.asarray(hierarchy_audit["normalization_row_bounds"])[test]
    hierarchy_to_dense_regret_bound = float(
        np.mean(np.asarray(chain_coefficients) * alpha * row_bounds)
    )
    hierarchy_to_dense_regret_difference = float(
        np.mean(np.abs(learned_regret_rows - dense_regret_rows))
    )
    return {
        "fold": int(fold),
        "train_instances": int(len(train)),
        "test_instances": int(len(test)),
        "development_instances": int(len(development)),
        "baseline_loss": baseline_loss,
        "ridge_loss": ridge_loss,
        "knn_loss": knn_loss,
        "tie_knn_loss": tie_knn_loss,
        "kernel_ridge_loss": kernel_ridge_loss,
        "dense_reference_loss": dense_reference_loss,
        "learned_loss": learned_loss,
        "rcol_selected_alpha": alpha,
        "rcol_test_rank_counterfactual_term": rank_counterfactual_term,
        "rcol_test_total_loss": float(rcol_test_loss),
        "baseline_task_error": _qaoa_regret(
            baseline, test, optima, edge_sets
        ),
        "ridge_task_error": _qaoa_regret(
            ridge_prediction, test, optima, edge_sets
        ),
        "knn_task_error": _qaoa_regret(
            knn_prediction, test, optima, edge_sets
        ),
        "tie_knn_task_error": _qaoa_regret(
            tie_knn_prediction, test, optima, edge_sets
        ),
        "tie_knn_qaoa_bound_mean": float(np.mean(tie_knn_bounds)),
        "tie_knn_qaoa_bound_max_ratio": float(
            np.max(tie_knn_regret_rows / np.maximum(tie_knn_bounds, 1e-15))
        ),
        "kernel_ridge_task_error": _qaoa_regret(
            kernel_ridge_prediction, test, optima, edge_sets
        ),
        "dense_reference_task_error": _qaoa_regret(
            dense_prediction, test, optima, edge_sets
        ),
        "learned_task_error": _qaoa_regret(
            prediction, test, optima, edge_sets
        ),
        "dense_prediction_relative_error": dense_prediction_error,
        "ridge_regularization": float(ridge_regularization),
        "knn_neighbors": int(knn_neighbors),
        "tie_knn_neighbors": int(tie_knn_neighbors),
        "tie_knn_effective_neighbors": tie_knn_effective_neighbors,
        "kernel_ridge_bandwidth": float(kernel_ridge_bandwidth),
        "kernel_ridge_regularization": float(kernel_ridge_regularization),
        "qaoa_regret_bound_mean": float(np.mean(regret_bounds)),
        "qaoa_regret_bound_max_ratio": float(
            np.max(learned_regret_rows / np.maximum(regret_bounds, 1e-15))
        ),
        "hierarchy_to_dense_regret_difference": (
            hierarchy_to_dense_regret_difference
        ),
        "hierarchy_to_dense_regret_bound": hierarchy_to_dense_regret_bound,
        "full_operator_frobenius_relative_error": float(
            hierarchy_audit["full_operator_frobenius_relative_error"]
        ),
        "full_operator_spectral_relative_error": float(
            hierarchy_audit["full_operator_spectral_relative_error"]
        ),
        "normalization_bound_frobenius": float(
            hierarchy_audit["normalization_bound_frobenius"]
        ),
        "normalization_bound_observed_coverage": bool(
            hierarchy_audit["normalization_bound_observed_coverage"]
        ),
        "observed_two_rank_contraction_ratio": float(
            hierarchy_audit["observed_two_rank_contraction_ratio"]
        ),
        "two_rank_condition_observed": bool(
            hierarchy_audit["two_rank_condition_observed"]
        ),
        "two_rank_bound_observed_coverage": bool(
            hierarchy_audit["two_rank_bound_observed_coverage"]
        ),
        "selected": selected,
        "prior": prior,
        "standardized": standardized,
        "fit_mask": fit_mask,
        "train_mask": train_mask,
        "test": test,
    }


def _evaluate_signature_group_fold(
    features: np.ndarray,
    targets: np.ndarray,
    optima: np.ndarray,
    edge_sets: list[list[tuple[int, int]]],
    groups: np.ndarray,
    fold: int,
) -> dict:
    """Evaluate neighbors with complete feature signatures held out."""

    indices = np.arange(len(features))
    test = indices[groups % 4 == fold]
    train = indices[groups % 4 != fold]
    train_groups = np.unique(groups[train])
    development_groups = train_groups[
        np.arange(len(train_groups)) % 5 == fold % 5
    ]
    development = train[np.isin(groups[train], development_groups)]
    fit = np.setdiff1d(train, development)
    if len(development) == 0 or len(fit) == 0 or len(test) == 0:
        raise AssertionError("signature-group fold is empty")
    mean, std = features[train].mean(axis=0), features[train].std(axis=0)
    std[std < 1e-9] = 1.0
    standardized = (features - mean) / std
    _, knn_prediction, knn_neighbors = _knn_prediction(
        standardized, targets, fit, development, train, test
    )
    _, tie_prediction, tie_neighbors, effective_neighbors = (
        _tie_complete_knn_prediction(
            standardized, targets, fit, development, train, test
        )
    )
    constant = np.tile(targets[train].mean(axis=0), (len(test), 1))
    knn_prediction = _clip_angles(knn_prediction)
    tie_prediction = _clip_angles(tie_prediction)
    tie_regret_rows = _qaoa_regret_rows(
        tie_prediction, test, optima, edge_sets
    )
    tie_bounds = _qaoa_angle_regret_bounds(
        tie_prediction, test, targets, optima, edge_sets
    )
    return {
        "fold": int(fold),
        "train_instances": int(len(train)),
        "test_instances": int(len(test)),
        "development_instances": int(len(development)),
        "signature_overlap": int(len(set(groups[train]) & set(groups[test]))),
        "constant_loss": float(np.mean((constant - targets[test]) ** 2)),
        "constant_task_error": _qaoa_regret(
            constant, test, optima, edge_sets
        ),
        "knn_loss": float(np.mean((knn_prediction - targets[test]) ** 2)),
        "knn_task_error": _qaoa_regret(
            knn_prediction, test, optima, edge_sets
        ),
        "knn_neighbors": int(knn_neighbors),
        "tie_knn_loss": float(
            np.mean((tie_prediction - targets[test]) ** 2)
        ),
        "tie_knn_task_error": _qaoa_regret(
            tie_prediction, test, optima, edge_sets
        ),
        "tie_knn_qaoa_bound_mean": float(np.mean(tie_bounds)),
        "tie_knn_qaoa_bound_max_ratio": float(
            np.max(tie_regret_rows / np.maximum(tie_bounds, 1e-15))
        ),
        "tie_knn_neighbors": int(tie_neighbors),
        "tie_knn_effective_neighbors": effective_neighbors,
    }


def run_qaoa(
    data_path: Path | None = None,
    scaling_data_path: Path | None = None,
) -> dict:
    path = fetch_gset_g1(data_path)
    n, weighted_edges = parse_gset(path)
    adjacency = [[] for _ in range(n)]
    for u, v, _ in weighted_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    features: list[np.ndarray] = []
    targets: list[list[float]] = []
    optima: list[float] = []
    edge_sets: list[list[tuple[int, int]]] = []
    vertex_sets: list[set[int]] = []
    gamma_grid = np.linspace(0.08, math.pi - 0.08, 9)
    beta_grid = np.linspace(0.04, math.pi / 2 - 0.04, 9)
    for sample in range(48):
        vertices = bfs_subgraph(adjacency, (sample * 37) % n, 10)
        index = {vertex: i for i, vertex in enumerate(vertices)}
        local = [
            (index[u], index[v])
            for u in vertices
            for v in adjacency[u]
            if u < v and v in index
        ]
        if not local:
            continue
        best = (-1.0, 0.0, 0.0)
        for gamma in gamma_grid:
            for beta in beta_grid:
                value = qaoa_energy(
                    local, float(gamma), float(beta), len(vertices)
                )
                if value > best[0]:
                    best = (value, float(gamma), float(beta))
        features.append(graph_features(local, len(vertices)))
        targets.append([best[1], best[2]])
        optima.append(best[0])
        edge_sets.append(local)
        vertex_sets.append(set(vertices))

    feature_array = np.asarray(features)
    target_array = np.asarray(targets)
    optimum_array = np.asarray(optima)
    pairwise_overlaps = [
        len(vertex_sets[i] & vertex_sets[j]) / len(vertex_sets[i] | vertex_sets[j])
        for i in range(len(vertex_sets))
        for j in range(i + 1, len(vertex_sets))
    ]
    folds = [
        _evaluate_qaoa_fold(
            feature_array, target_array, optimum_array, edge_sets, fold
        )
        for fold in range(4)
    ]
    _, signature_groups = np.unique(
        feature_array, axis=0, return_inverse=True
    )
    signature_folds = [
        _evaluate_signature_group_fold(
            feature_array,
            target_array,
            optimum_array,
            edge_sets,
            signature_groups,
            fold,
        )
        for fold in range(4)
    ]
    main = folds[0]

    # Runtime scaling on prefixes of the actual fold-zero training set.  The
    # result is a timing diagnostic only; its numerical output is not used for
    # model selection or scientific scoring.
    training_indices = np.flatnonzero(main["train_mask"])
    query_indices = np.asarray(main["test"], dtype=int)
    tie_runtime = []
    for reference_count in (8, 16, 24, len(training_indices)):
        references = training_indices[:reference_count]
        samples = []
        for _ in range(101):
            started = time.perf_counter()
            _tie_complete_prediction(
                main["standardized"][query_indices],
                main["standardized"][references],
                target_array[references],
                1,
            )
            samples.append(time.perf_counter() - started)
        tie_runtime.append(
            {
                "reference_records": int(reference_count),
                "query_records": int(len(query_indices)),
                "feature_dimension": int(feature_array.shape[1]),
                "median_seconds": float(np.median(samples)),
                "distance_products": int(
                    len(query_indices) * reference_count * feature_array.shape[1]
                ),
            }
        )

    dense_prior = dense_reference_prior(
        main["standardized"],
        main["train_mask"],
        target_array,
        bandwidth=0.10,
    )
    rank_frontier = []
    for rank in (2, 3, 4, 6, 8):
        candidate = hierarchical_prior_bundle(
            main["standardized"],
            [main["fit_mask"], main["train_mask"]],
            target_array,
            CONFIG["seed"],
            bandwidth=0.10,
            leaf_size=8,
            low_rank=2,
            high_rank=rank,
        )
        prediction_error = float(
            np.linalg.norm(candidate["high"][1] - dense_prior)
            / max(float(np.linalg.norm(dense_prior)), 1e-15)
        )
        dense_audit = dense_hierarchy_audit(
            main["standardized"],
            main["train_mask"],
            target_array,
            CONFIG["seed"],
            bandwidth=0.10,
            leaf_size=8,
            low_rank=2,
            high_rank=rank,
        )
        rank_frontier.append(
            {
                "rank": int(rank),
                "sampled_relative_error": float(
                    candidate["metrics"]["hmat_sampled_row_relative_error"]
                ),
                "prediction_relative_error": prediction_error,
                "full_operator_frobenius_relative_error": float(
                    dense_audit["full_operator_frobenius_relative_error"]
                ),
                "full_operator_spectral_relative_error": float(
                    dense_audit["full_operator_spectral_relative_error"]
                ),
                "storage_fraction": float(
                    candidate["metrics"]["hmat_storage_fraction"]
                ),
                "counted_work_ratio": float(
                    candidate["metrics"]["hmat_matmat_work_ratio"]
                ),
            }
        )

    degrees = np.array([len(vertices) for vertices in adjacency], dtype=float)
    scaling_features = np.column_stack((degrees, np.arange(n) / n))
    scaling_frontier = []
    hierarchy = None
    for rank in (2, 3, 4, 6, 8):
        candidate = matrix_free_certificate(
            scaling_features,
            CONFIG["seed"],
            bandwidth=0.08,
            leaf_size=32,
            low_rank=2,
            high_rank=rank,
        )
        scaling_frontier.append(
            {
                "rank": int(rank),
                "sampled_relative_error": float(
                    candidate["hmat_sampled_row_relative_error"]
                ),
                "storage_fraction": float(candidate["hmat_storage_fraction"]),
                "counted_work_ratio": float(
                    candidate["hmat_matmat_work_ratio"]
                ),
                "queried_entry_fraction": float(
                    candidate["hmat_entry_query_fraction"]
                ),
            }
        )
        if rank == 8:
            hierarchy = candidate
    if hierarchy is None:
        raise RuntimeError("registered scaling rank was not evaluated")

    g81_path = fetch_gset_g81(scaling_data_path)
    g81_n, g81_edges = parse_gset(g81_path)
    g81_degrees = np.zeros(g81_n, dtype=float)
    for u, v, _ in g81_edges:
        g81_degrees[u] += 1.0
        g81_degrees[v] += 1.0
    g81_features = np.column_stack((g81_degrees, np.arange(g81_n) / g81_n))
    g81 = matrix_free_certificate(
        g81_features,
        CONFIG["seed"],
        bandwidth=0.08,
        leaf_size=32,
        low_rank=2,
        high_rank=8,
    )
    g81_metrics = {f"g81_{key}": value for key, value in g81.items()}
    runtime_audit = [
        fair_dense_runtime_audit(
            g81_features[:size],
            CONFIG["seed"] + size,
            bandwidth=0.08,
            leaf_size=32,
            rank=8,
            right_hand_sides=4,
            repeats=7,
        )
        for size in (256, 512, 1024, 2048)
    ]

    public_folds = [
        {
            key: value
            for key, value in fold.items()
            if key
            not in {
                "selected",
                "prior",
                "standardized",
                "fit_mask",
                "train_mask",
                "test",
            }
        }
        for fold in folds
    ]
    fold_losses = np.array([fold["learned_loss"] for fold in folds])
    fold_baselines = np.array([fold["baseline_loss"] for fold in folds])
    fold_task_errors = np.array(
        [fold["learned_task_error"] for fold in folds]
    )
    selected = main["selected"]
    prior_metrics = {
        f"prior_{key}": value for key, value in main["prior"]["metrics"].items()
    }
    return {
        "dataset": "Stanford Gset G1",
        "dataset_url": GSET_URL,
        "dataset_sha256": GSET_SHA256,
        "records": n,
        "edges": len(weighted_edges),
        "g81_dataset": "Stanford Gset G81",
        "g81_dataset_url": GSET_G81_URL,
        "g81_dataset_sha256": GSET_G81_SHA256,
        "g81_records": g81_n,
        "g81_edges": len(g81_edges),
        "g81_timing_platform": platform.platform(),
        "learning_instances": len(feature_array),
        "qaoa_depth": 1,
        "qaoa_grid_points_per_instance": int(len(gamma_grid) * len(beta_grid)),
        "unique_angle_grid_labels": int(len(np.unique(target_array, axis=0))),
        "unique_feature_vectors": int(len(np.unique(feature_array, axis=0))),
        "mean_pairwise_vertex_jaccard": float(np.mean(pairwise_overlaps)),
        "maximum_pairwise_vertex_jaccard": float(np.max(pairwise_overlaps)),
        "g1_unique_edge_weights": sorted(
            {float(weight) for _, _, weight in weighted_edges}
        ),
        "g81_unique_edge_weights": sorted(
            {float(weight) for _, _, weight in g81_edges}
        ),
        "train_instances": main["train_instances"],
        "test_instances": main["test_instances"],
        "development_instances": main["development_instances"],
        "feature_dimension": feature_array.shape[1],
        "target_kind": "QAOA angles",
        "baseline_loss": main["baseline_loss"],
        "ridge_loss": main["ridge_loss"],
        "knn_loss": main["knn_loss"],
        "tie_knn_loss": main["tie_knn_loss"],
        "kernel_ridge_loss": main["kernel_ridge_loss"],
        "dense_reference_loss": main["dense_reference_loss"],
        "learned_loss": main["learned_loss"],
        "rcol_selected_alpha": main["rcol_selected_alpha"],
        "rcol_consistency_weight": selected["consistency_weight"],
        "rcol_development_data_loss": selected["data_loss"],
        "rcol_development_rank_counterfactual_term": selected[
            "rank_counterfactual_term"
        ],
        "rcol_certificate_violation": selected["certificate_violation"],
        "rcol_budget_violation": selected["budget_violation"],
        "rcol_error_tolerance": selected["error_tolerance"],
        "rcol_storage_budget": selected["storage_budget"],
        "rcol_development_total_loss": selected["total_loss"],
        "rcol_test_rank_counterfactual_term": main[
            "rcol_test_rank_counterfactual_term"
        ],
        "rcol_test_total_loss": main["rcol_test_total_loss"],
        "rcol_alpha_trace": selected["trace"],
        "baseline_task_error": main["baseline_task_error"],
        "ridge_task_error": main["ridge_task_error"],
        "knn_task_error": main["knn_task_error"],
        "tie_knn_task_error": main["tie_knn_task_error"],
        "kernel_ridge_task_error": main["kernel_ridge_task_error"],
        "dense_reference_task_error": main[
            "dense_reference_task_error"
        ],
        "learned_task_error": main["learned_task_error"],
        "dense_prediction_relative_error": main[
            "dense_prediction_relative_error"
        ],
        "ridge_regularization": main["ridge_regularization"],
        "knn_neighbors": main["knn_neighbors"],
        "tie_knn_neighbors": main["tie_knn_neighbors"],
        "tie_knn_effective_neighbors": main["tie_knn_effective_neighbors"],
        "kernel_ridge_bandwidth": main["kernel_ridge_bandwidth"],
        "kernel_ridge_regularization": main[
            "kernel_ridge_regularization"
        ],
        "cross_validation_folds": 4,
        "cross_validation": public_folds,
        "signature_group_cross_validation": signature_folds,
        "signature_group_count": int(len(np.unique(signature_groups))),
        "signature_label_conflicts": int(
            sum(
                len(np.unique(target_array[signature_groups == group], axis=0)) > 1
                for group in np.unique(signature_groups)
            )
        ),
        "signature_group_overlap_total": int(
            sum(row["signature_overlap"] for row in signature_folds)
        ),
        "tie_knn_runtime": tie_runtime,
        "cross_validation_learned_loss_mean": float(fold_losses.mean()),
        "cross_validation_learned_loss_std": float(
            fold_losses.std(ddof=1)
        ),
        "cross_validation_baseline_loss_mean": float(
            fold_baselines.mean()
        ),
        "cross_validation_baseline_loss_std": float(
            fold_baselines.std(ddof=1)
        ),
        "cross_validation_task_error_mean": float(
            fold_task_errors.mean()
        ),
        "cross_validation_task_error_std": float(
            fold_task_errors.std(ddof=1)
        ),
        "runtime_audit": runtime_audit,
        "rank_frontier": rank_frontier,
        "scaling_frontier": scaling_frontier,
        **prior_metrics,
        **hierarchy,
        **g81_metrics,
    }


def syndrome_features(value: str) -> np.ndarray:
    arr = np.asarray(ast.literal_eval(value), dtype=float).reshape(4, 4)
    flat = arr.ravel()
    temporal = np.abs(np.diff(arr, axis=0)).ravel()
    round_parity = np.mod(arr.sum(axis=1), 2)
    spatial_parity = np.mod(arr.sum(axis=0), 2)
    return np.concatenate((flat, temporal, round_parity, spatial_parity, [flat.sum()]))


def balanced_weights(y: np.ndarray, quantity: np.ndarray) -> np.ndarray:
    out = quantity.astype(float).copy()
    for label in (0.0, 1.0):
        mask = y == label
        out[mask] *= 0.5 / max(out[mask].sum(), 1.0)
    return out


def logistic_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    for _ in range(30):
        score = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-score))
        grad = x.T @ (weights * (p - y)) + 1e-4 * (eye @ beta)
        curvature = weights * p * (1 - p)
        hess = (x.T * curvature) @ x + 1e-4 * eye + 1e-9 * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.linalg.norm(step) < 1e-10:
            break
    return beta


def run_qec() -> dict:
    path = PACKAGE_ROOT / "data" / "source" / "syndromes_dataset.zip"
    expected = "bdfce36a71f04295ac78fb372d9c2e381801c05e3be119f919750ef59026d072"
    if sha256(path) != expected:
        raise ValueError("QEC dataset SHA-256 mismatch")
    rows = []
    rates = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        for rate_index, name in enumerate(names):
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    rows.append(
                        (
                            syndrome_features(row["syndromes"]),
                            float(row["labels"]),
                            float(row["quantity"]),
                        )
                    )
                    rates.append(rate_index)
    x = np.vstack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    quantity = np.array([r[2] for r in rows])
    rates = np.array(rates)
    split = CONFIG.get("split", "rate")
    indices = np.arange(len(x))
    if split == "hash":
        test_mask = indices % 5 == 0
    elif split == "alternating":
        test_mask = rates % 2 == 1
    elif split == "middle":
        test_mask = rates == 4
    else:
        test_mask = rates >= 5
    train_mask = ~test_mask
    mean = np.average(x[train_mask], axis=0, weights=quantity[train_mask])
    var = np.average((x[train_mask] - mean) ** 2, axis=0, weights=quantity[train_mask])
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    design = np.column_stack((np.ones(len(z)), z))
    train_rates = np.unique(rates[train_mask])
    development_rate = int(train_rates[-1])
    development_mask = train_mask & (rates == development_rate)
    fit_mask = train_mask & ~development_mask
    fit_weights = balanced_weights(y[fit_mask], quantity[fit_mask])
    beta_fit = logistic_fit(design[fit_mask], y[fit_mask], fit_weights)
    base_development = 1.0 / (
        1.0
        + np.exp(
            -np.clip(design[development_mask] @ beta_fit, -30, 30)
        )
    )
    train_weights = balanced_weights(y[train_mask], quantity[train_mask])
    beta = logistic_fit(design[train_mask], y[train_mask], train_weights)
    test_weights = balanced_weights(y[test_mask], quantity[test_mask])
    base_test = 1.0 / (
        1.0 + np.exp(-np.clip(design[test_mask] @ beta, -30, 30))
    )
    rate_feature = rates[:, None] / max(float(rates.max()), 1.0)
    prior = hierarchical_prior_bundle(
        np.column_stack((z, rate_feature)),
        [fit_mask, train_mask],
        y,
        CONFIG["seed"],
        bandwidth=0.10,
        leaf_size=64,
        low_rank=2,
        high_rank=8,
    )
    development_weights = balanced_weights(
        y[development_mask], quantity[development_mask]
    )
    selected = select_rank_counterfactual_blend(
        y[development_mask],
        base_development,
        np.clip(prior["low"][0][development_mask, 0], 1e-8, 1.0 - 1e-8),
        np.clip(prior["high"][0][development_mask, 0], 1e-8, 1.0 - 1e-8),
        weights=development_weights,
        loss="log",
        sampled_operator_error=prior["metrics"][
            "hmat_sampled_row_relative_error"
        ],
        storage_fraction=prior["metrics"]["hmat_storage_fraction"],
    )
    alpha = selected["alpha"]
    p = (1.0 - alpha) * base_test + alpha * np.clip(
        prior["high"][1][test_mask, 0], 1e-8, 1.0 - 1e-8
    )
    p_low = (1.0 - alpha) * base_test + alpha * np.clip(
        prior["low"][1][test_mask, 0], 1e-8, 1.0 - 1e-8
    )
    yt = y[test_mask]
    eps = 1e-12
    logistic_only_loss = float(
        np.sum(
            test_weights
            * (
                -(
                    yt * np.log(base_test + eps)
                    + (1 - yt) * np.log(1 - base_test + eps)
                )
            )
        )
    )
    learned_loss = float(
        np.sum(test_weights * (-(yt * np.log(p + eps) + (1 - yt) * np.log(1 - p + eps))))
    )
    baseline_loss = float(-math.log(0.5))
    logistic_only_brier = float(np.sum(test_weights * (base_test - yt) ** 2))
    learned_brier = float(np.sum(test_weights * (p - yt) ** 2))
    baseline_brier = 0.25
    accuracy = float(np.sum(test_weights * ((p >= 0.5) == yt)))
    target_scale = max(
        float(np.sum(test_weights * (yt - np.sum(test_weights * yt)) ** 2)),
        1e-8,
    )
    rank_counterfactual_term = float(
        np.sum(test_weights * (p - p_low) ** 2) / target_scale
    )
    rcol_test_loss = (
        learned_loss
        + selected["consistency_weight"] * rank_counterfactual_term
        + selected["certificate_weight"] * selected["certificate_violation"]
        + selected["budget_weight"] * selected["budget_violation"]
    )
    return {
        "dataset": "Open Zenodo surface-code syndrome counts (distance 3)",
        "dataset_url": "https://doi.org/10.5281/zenodo.11166209",
        "dataset_sha256": expected,
        "records": int(quantity.sum()),
        "unique_weighted_rows": len(rows),
        "noise_rates": len(np.unique(rates)),
        "train_records": int(quantity[train_mask].sum()),
        "test_records": int(quantity[test_mask].sum()),
        "feature_dimension": design.shape[1],
        "baseline_loss": baseline_loss,
        "logistic_only_loss": logistic_only_loss,
        "learned_loss": learned_loss,
        "rcol_selected_alpha": alpha,
        "rcol_consistency_weight": selected["consistency_weight"],
        "rcol_development_data_loss": selected["data_loss"],
        "rcol_development_rank_counterfactual_term": selected[
            "rank_counterfactual_term"
        ],
        "rcol_certificate_violation": selected["certificate_violation"],
        "rcol_budget_violation": selected["budget_violation"],
        "rcol_error_tolerance": selected["error_tolerance"],
        "rcol_storage_budget": selected["storage_budget"],
        "rcol_development_total_loss": selected["total_loss"],
        "rcol_test_rank_counterfactual_term": rank_counterfactual_term,
        "rcol_test_total_loss": rcol_test_loss,
        "baseline_task_error": baseline_brier,
        "logistic_only_task_error": logistic_only_brier,
        "learned_task_error": learned_brier,
        "balanced_accuracy": accuracy,
        **prior["metrics"],
    }


def reproduce(
    data_path: Path | None = None,
    scaling_data_path: Path | None = None,
) -> dict:
    if CONFIG["domain"] == "qaoa":
        result = run_qaoa(data_path, scaling_data_path)
    elif CONFIG["domain"] == "qec":
        result = run_qec()
    else:
        raise ValueError("domain must be 'qaoa' or 'qec'")
    return {
        "schema_version": 2,
        "package": CONFIG["package"],
        "manuscript_key": CONFIG["key"],
        "seed": CONFIG["seed"],
        "task": CONFIG["task"],
        **result,
    }


def _rounded_value(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _rounded_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_value(item) for item in value]
    return value


def rounded(result: dict) -> dict:
    return _rounded_value(result)


def write_outputs(result: dict) -> None:
    """Write one canonical JSON, one macro file, and four evidence figures."""

    result = rounded(result)
    outputs = [PACKAGE_ROOT / "results"]
    if SOURCE_ROOT != PACKAGE_ROOT:
        outputs.append(SOURCE_ROOT / "results")
    canonical_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(canonical_json)
    (PACKAGE_ROOT / "results" / "locked_results.json").write_text(canonical_json)

    folds = result["cross_validation"]
    methods = ("baseline", "ridge", "knn", "tie_knn", "kernel_ridge", "learned")
    summary: dict[str, float] = {}
    for method in methods:
        for metric in ("loss", "task_error"):
            values = np.asarray([row[f"{method}_{metric}"] for row in folds])
            summary[f"{method}_{metric}_mean"] = float(values.mean())
            summary[f"{method}_{metric}_std"] = float(values.std(ddof=1))
    signature_summary: dict[str, float] = {}
    for method in ("constant", "knn", "tie_knn"):
        for metric in ("loss", "task_error"):
            values = np.asarray(
                [
                    row[f"{method}_{metric}"]
                    for row in result["signature_group_cross_validation"]
                ]
            )
            signature_summary[f"{method}_{metric}_mean"] = float(values.mean())
    rank_four = next(row for row in result["rank_frontier"] if row["rank"] == 4)
    commands = {
        "LearningTasks": str(result["learning_instances"]),
        "UniqueLabels": str(result["unique_angle_grid_labels"]),
        "UniqueFeatures": str(result["unique_feature_vectors"]),
        "GOneVertices": f'{int(result["records"]):,}',
        "GOneEdges": f'{int(result["edges"]):,}',
        "GEightyOneVertices": f'{int(result["g81_records"]):,}',
        "GEightyOneEdges": f'{int(result["g81_edges"]):,}',
        "MeanConstantMSE": f'{summary["baseline_loss_mean"]:.5f}',
        "MeanRidgeMSE": f'{summary["ridge_loss_mean"]:.5f}',
        "MeanKNNMSE": f'{summary["knn_loss_mean"]:.5f}',
        "MeanTieKNNMSE": f'{summary["tie_knn_loss_mean"]:.5f}',
        "MeanKRRMSE": f'{summary["kernel_ridge_loss_mean"]:.5f}',
        "MeanBlendMSE": f'{summary["learned_loss_mean"]:.5f}',
        "MeanConstantRegret": f'{summary["baseline_task_error_mean"]:.5f}',
        "MeanRidgeRegret": f'{summary["ridge_task_error_mean"]:.5f}',
        "MeanKNNRegret": f'{summary["knn_task_error_mean"]:.5f}',
        "MeanTieKNNRegret": f'{summary["tie_knn_task_error_mean"]:.5f}',
        "MeanKRRRegret": f'{summary["kernel_ridge_task_error_mean"]:.5f}',
        "MeanBlendRegret": f'{summary["learned_task_error_mean"]:.5f}',
        "SignatureConstantMSE": f'{signature_summary["constant_loss_mean"]:.5f}',
        "SignatureKNNMSE": f'{signature_summary["knn_loss_mean"]:.5f}',
        "SignatureTieKNNMSE": f'{signature_summary["tie_knn_loss_mean"]:.5f}',
        "SignatureConstantRegret": f'{signature_summary["constant_task_error_mean"]:.5f}',
        "SignatureKNNRegret": f'{signature_summary["knn_task_error_mean"]:.5f}',
        "SignatureTieKNNRegret": f'{signature_summary["tie_knn_task_error_mean"]:.5f}',
        "RankFourSampledError": f'{rank_four["sampled_relative_error"]:.3e}',
        "RankFourFullError": f'{rank_four["full_operator_frobenius_relative_error"]:.3e}',
        "RankFourPredictionError": f'{rank_four["prediction_relative_error"]:.3e}',
        "RankFourStorage": f'{100 * rank_four["storage_fraction"]:.1f}',
        "RankFourWorkFraction": f'{100 / rank_four["counted_work_ratio"]:.1f}',
        "GEightyOneSampledError": f'{result["g81_hmat_sampled_row_relative_error"]:.3e}',
        "GEightyOneStorage": f'{100 * result["g81_hmat_storage_fraction"]:.3f}',
        "GEightyOneWorkFraction": f'{100 / result["g81_hmat_matmat_work_ratio"]:.3f}',
        "MaximumContractionRatio": f'{max(row["observed_two_rank_contraction_ratio"] for row in folds):.4f}',
        "MaximumRegretBoundRatio": f'{max(row["qaoa_regret_bound_max_ratio"] for row in folds):.4f}',
    }
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in commands.items()]
    for output in outputs:
        (output / "results.tex").write_text("\n".join(lines) + "\n")
        for stale_name in (
            "rank_rows.tex",
            "scaling_rows.tex",
            "fold_rows.tex",
            "alpha_rows.tex",
        ):
            (output / stale_name).unlink(missing_ok=True)

    from .figures import write_figures

    write_figures(result, outputs[0] / "figures")
    for output in outputs[1:]:
        destination = output / "figures"
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.pdf"):
            stale.unlink()
        for figure in (outputs[0] / "figures").glob("*.pdf"):
            shutil.copy2(figure, destination / figure.name)


def verify(result: dict) -> None:
    expected_path = PACKAGE_ROOT / "results" / "results.json"
    if not expected_path.exists():
        raise SystemExit("No committed results.json; run with --write first")
    expected = json.loads(expected_path.read_text())
    actual = rounded(result)
    def compare(value, old, path: str) -> None:
        key = path.rsplit(".", 1)[-1]
        if "seconds" in key or key.endswith("platform"):
            return
        if isinstance(value, dict):
            if not isinstance(old, dict) or value.keys() != old.keys():
                raise SystemExit(f"Result key mismatch at {path}")
            for child, item in value.items():
                compare(item, old[child], f"{path}.{child}")
            return
        if isinstance(value, list):
            if not isinstance(old, list) or len(value) != len(old):
                raise SystemExit(f"Result list mismatch at {path}")
            for index, item in enumerate(value):
                compare(item, old[index], f"{path}[{index}]")
            return
        if isinstance(value, float):
            if not math.isclose(value, old, rel_tol=1e-9, abs_tol=1e-10):
                raise SystemExit(f"Mismatch at {path}: expected {old}, got {value}")
            return
        if value != old:
            raise SystemExit(f"Mismatch at {path}: expected {old}, got {value}")

    compare(actual, expected, "results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and verify the manuscript evidence")
    parser.add_argument("--write", action="store_true", help="write locked JSON and TeX result artifacts")
    parser.add_argument("--verify", action="store_true", help="compare a fresh run with locked results")
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="fetch and authenticate Stanford Gset G1 and G81 without running the experiment",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="explicit G1 cache path (or set HQML_G1_PATH)",
    )
    parser.add_argument(
        "--scaling-data-path",
        type=Path,
        help="explicit G81 cache path (or set HQML_G81_PATH)",
    )
    args = parser.parse_args()
    if args.fetch_data:
        print(fetch_gset_g1(args.data_path))
        print(fetch_gset_g81(args.scaling_data_path))
        if not args.write and not args.verify:
            return
    result = reproduce(args.data_path, args.scaling_data_path)
    if args.write:
        write_outputs(result)
    if args.verify:
        verify(result)
    print(json.dumps(rounded(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
