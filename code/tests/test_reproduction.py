from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hqml_lie_residual import experiment


class ReproductionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = experiment.reproduce()
        cls.actual = experiment.rounded(cls.raw)

    def test_locked_results_replay(self) -> None:
        experiment.verify(self.raw)
        expected = json.loads(
            (Path(experiment.PACKAGE_ROOT) / "results" / "results.json").read_text()
        )
        self.assertEqual(expected["schema_version"], 2)
        self.assertEqual(expected.keys(), self.actual.keys())

    def test_authenticated_public_inputs(self) -> None:
        g1 = experiment.fetch_gset_g1()
        g81 = experiment.fetch_gset_g81()
        self.assertEqual(experiment.sha256(g1), experiment.GSET_SHA256)
        self.assertEqual(experiment.sha256(g81), experiment.GSET_G81_SHA256)
        self.assertEqual(experiment.parse_gset(g1)[0], 800)
        self.assertEqual(experiment.parse_gset(g81)[0], 20_000)

    def test_stronger_baselines_turn_the_headline_negative(self) -> None:
        folds = self.raw["cross_validation"]
        knn_mse = np.mean([row["knn_loss"] for row in folds])
        blend_mse = np.mean([row["learned_loss"] for row in folds])
        knn_regret = np.mean([row["knn_task_error"] for row in folds])
        blend_regret = np.mean([row["learned_task_error"] for row in folds])
        self.assertLess(knn_mse, blend_mse)
        self.assertLess(knn_regret, blend_regret)
        self.assertTrue(
            any(row["learned_loss"] > row["ridge_loss"] for row in folds)
        )
        self.assertTrue(any(row["rcol_selected_alpha"] == 0.0 for row in folds))

    def test_tie_complete_prediction_is_permutation_invariant(self) -> None:
        reference = np.asarray([[0.0], [0.0], [1.0], [2.0]])
        targets = np.asarray([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0]])
        query = np.asarray([[0.0]])
        expected, counts = experiment._tie_complete_prediction(
            query, reference, targets, 1
        )
        permutation = np.asarray([3, 1, 0, 2])
        actual, permuted_counts = experiment._tie_complete_prediction(
            query, reference[permutation], targets[permutation], 1
        )
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(permuted_counts, counts)

    def test_signature_protocol_has_zero_overlap(self) -> None:
        self.assertEqual(self.raw["signature_group_overlap_total"], 0)
        self.assertEqual(self.raw["signature_label_conflicts"], 0)

    def test_all_audited_bounds_cover_but_are_conditional(self) -> None:
        for row in self.raw["cross_validation"]:
            self.assertTrue(row["normalization_bound_observed_coverage"])
            self.assertTrue(row["two_rank_condition_observed"])
            self.assertTrue(row["two_rank_bound_observed_coverage"])
            self.assertLessEqual(row["qaoa_regret_bound_max_ratio"], 1.0 + 1e-12)
            self.assertLessEqual(
                row["hierarchy_to_dense_regret_difference"],
                row["hierarchy_to_dense_regret_bound"] + 1e-12,
            )
        # A sampled action diagnostic is not an operator-norm upper bound.
        self.assertTrue(
            any(
                row["sampled_relative_error"]
                < row["full_operator_frobenius_relative_error"]
                for row in self.raw["rank_frontier"]
            )
        )

    def test_runtime_audit_compares_optimized_dense_apply(self) -> None:
        rows = self.raw["runtime_audit"]
        self.assertEqual([row["records"] for row in rows], [256, 512, 1024, 2048])
        for row in rows:
            self.assertGreater(row["hierarchy_apply_seconds"], 0.0)
            self.assertGreater(row["dense_apply_seconds"], 0.0)
            self.assertLess(row["full_relative_error"], 5e-4)
        # Preserve the registered snapshot without making a noisy live timing
        # ordering part of numerical replay.
        locked = json.loads(
            (Path(experiment.PACKAGE_ROOT) / "results" / "results.json").read_text()
        )
        self.assertTrue(
            all(
                row["dense_apply_seconds"] < row["hierarchy_apply_seconds"]
                for row in locked["runtime_audit"]
            )
        )

    def test_parser_rejects_invalid_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-gset"
            path.write_text("2 1\n1 1 1\n")
            with self.assertRaisesRegex(ValueError, "invalid endpoint or self-loop"):
                experiment.parse_gset(path)


if __name__ == "__main__":
    unittest.main()
