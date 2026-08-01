"""Deterministic perturbation bounds used by the audit manuscript."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationAudit:
    row_bounds: np.ndarray
    actual_row_errors: np.ndarray
    covered: bool


def normalized_prediction_error_bound(
    exact_numerator: np.ndarray,
    exact_denominator: np.ndarray,
    approximate_numerator: np.ndarray,
    approximate_denominator: np.ndarray,
) -> NormalizationAudit:
    """Bound errors in rowwise ratios using nonzero observed denominators.

    For each row, the returned bound is

    ``||n-ntilde||/|d| + ||ntilde|| |d-dtilde|/(|d| |dtilde|)``.

    It is deterministic and exact in real arithmetic.  It is not a
    generalization bound and is infinite if either denominator vanishes.
    """

    numerator = np.asarray(exact_numerator, dtype=float)
    approximate = np.asarray(approximate_numerator, dtype=float)
    denominator = np.asarray(exact_denominator, dtype=float).reshape(-1)
    approximate_denominator = np.asarray(
        approximate_denominator, dtype=float
    ).reshape(-1)
    if numerator.ndim == 1:
        numerator = numerator[:, None]
    if approximate.ndim == 1:
        approximate = approximate[:, None]
    if numerator.shape != approximate.shape:
        raise ValueError("numerator shapes differ")
    if len(denominator) != len(numerator) or len(approximate_denominator) != len(
        numerator
    ):
        raise ValueError("denominator shapes differ")

    absolute_denominator = np.abs(denominator)
    absolute_approximate_denominator = np.abs(approximate_denominator)
    valid = (absolute_denominator > 0.0) & (
        absolute_approximate_denominator > 0.0
    )
    row_bounds = np.full(len(numerator), np.inf, dtype=float)
    row_bounds[valid] = (
        np.linalg.norm(numerator[valid] - approximate[valid], axis=1)
        / absolute_denominator[valid]
        + np.linalg.norm(approximate[valid], axis=1)
        * np.abs(denominator[valid] - approximate_denominator[valid])
        / (
            absolute_denominator[valid]
            * absolute_approximate_denominator[valid]
        )
    )
    exact_prediction = numerator / denominator[:, None]
    approximate_prediction = approximate / approximate_denominator[:, None]
    actual = np.linalg.norm(exact_prediction - approximate_prediction, axis=1)
    return NormalizationAudit(
        row_bounds=row_bounds,
        actual_row_errors=actual,
        covered=bool(np.all(actual <= row_bounds * (1.0 + 1e-12))),
    )


@dataclass(frozen=True)
class TwoRankAudit:
    contraction_ratio: float
    error_bound: float
    condition_observed: bool
    covered: bool


def two_rank_contraction_audit(
    low_rank_prediction: np.ndarray,
    high_rank_prediction: np.ndarray,
    reference_prediction: np.ndarray,
) -> TwoRankAudit:
    """Evaluate the conditional two-rank inequality on an audited batch."""

    low = np.asarray(low_rank_prediction, dtype=float)
    high = np.asarray(high_rank_prediction, dtype=float)
    reference = np.asarray(reference_prediction, dtype=float)
    low_error = float(np.linalg.norm(low - reference))
    high_error = float(np.linalg.norm(high - reference))
    if low_error == 0.0:
        ratio = 0.0 if high_error == 0.0 else np.inf
    else:
        ratio = high_error / low_error
    condition = bool(ratio < 1.0)
    bound = (
        ratio / (1.0 - ratio) * float(np.linalg.norm(high - low))
        if condition
        else np.inf
    )
    return TwoRankAudit(
        contraction_ratio=float(ratio),
        error_bound=float(bound),
        condition_observed=condition,
        covered=bool(condition and high_error <= bound * (1.0 + 1e-12)),
    )


def qaoa_depth_one_lipschitz_constant(
    absolute_edge_weight_sum: float,
    qubits: int,
) -> float:
    """Return a conservative Euclidean parameter Lipschitz constant.

    With ``||C|| <= sum |w_e|`` and ``||sum_j X_j|| <= qubits``, the two
    commutator bounds combine to this constant for depth-one QAOA.
    """

    cost_norm_bound = float(absolute_edge_weight_sum)
    if cost_norm_bound < 0.0 or qubits < 0:
        raise ValueError("norm bounds must be nonnegative")
    return float(
        2.0
        * cost_norm_bound
        * np.sqrt(cost_norm_bound * cost_norm_bound + qubits * qubits)
    )


def qaoa_regret_bound(
    prediction: np.ndarray,
    reference_angles: np.ndarray,
    reference_objective: float,
    absolute_edge_weight_sum: float,
    qubits: int,
) -> float:
    """Bound normalized objective loss from an angle displacement."""

    if reference_objective <= 0.0:
        raise ValueError("reference objective must be positive")
    return float(
        qaoa_depth_one_lipschitz_constant(absolute_edge_weight_sum, qubits)
        * np.linalg.norm(
            np.asarray(prediction, dtype=float)
            - np.asarray(reference_angles, dtype=float)
        )
        / reference_objective
    )
