from __future__ import annotations

import unittest

import numpy as np

from hqml_lie_residual.bounds import (
    normalized_prediction_error_bound,
    qaoa_regret_bound,
    two_rank_contraction_audit,
)
from hqml_lie_residual.experiment import qaoa_energy
from hqml_lie_residual.matrix_free import select_rank_counterfactual_blend


class BoundTests(unittest.TestCase):
    def test_normalization_bound_covers_adversarial_perturbations(self) -> None:
        rng = np.random.default_rng(7701)
        for _ in range(500):
            numerator = rng.normal(size=(7, 2))
            denominator = rng.uniform(0.05, 2.0, size=7)
            approximate_numerator = numerator + 0.1 * rng.normal(size=(7, 2))
            approximate_denominator = denominator + rng.uniform(-0.04, 0.04, 7)
            audit = normalized_prediction_error_bound(
                numerator,
                denominator,
                approximate_numerator,
                approximate_denominator,
            )
            self.assertTrue(audit.covered)

    def test_two_rank_statement_requires_contraction(self) -> None:
        reference = np.zeros(3)
        low = np.array([2.0, 0.0, 0.0])
        high = np.array([0.5, 0.0, 0.0])
        passing = two_rank_contraction_audit(low, high, reference)
        self.assertTrue(passing.condition_observed)
        self.assertTrue(passing.covered)

        failing = two_rank_contraction_audit(high, low, reference)
        self.assertFalse(failing.condition_observed)
        self.assertFalse(failing.covered)
        self.assertTrue(np.isinf(failing.error_bound))

    def test_failed_sampled_diagnostic_rejects_nonzero_prior(self) -> None:
        truth = np.array([[0.0], [1.0]])
        baseline = np.full_like(truth, 0.5)
        selected = select_rank_counterfactual_blend(
            truth,
            baseline,
            truth,
            truth,
            sampled_operator_error=0.2,
            error_tolerance=0.01,
            maximum_alpha=1.0,
        )
        self.assertEqual(selected["alpha"], 0.0)
        self.assertFalse(selected["operator_diagnostic_pass"])

    def test_depth_one_qaoa_regret_is_below_commutator_bound(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        reference = np.array([0.73, 0.41])
        objective = qaoa_energy(edges, *reference, 4)
        rng = np.random.default_rng(7702)
        for _ in range(250):
            prediction = reference + rng.normal(scale=0.05, size=2)
            achieved = qaoa_energy(edges, *prediction, 4)
            regret = max(0.0, (objective - achieved) / objective)
            bound = qaoa_regret_bound(
                prediction, reference, objective, len(edges), 4
            )
            self.assertLessEqual(regret, bound + 1e-12)


if __name__ == "__main__":
    unittest.main()
