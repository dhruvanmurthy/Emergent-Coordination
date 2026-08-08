import unittest

from coordination_metrics import (
    bootstrap_confidence_interval,
    estimate_mi_from_records,
    mutual_information,
    permutation_null_distribution,
    smoothing_sensitivity,
)


class CoordinationMetricsTests(unittest.TestCase):
    def test_perfect_mapping_has_one_bit_of_mi(self):
        sources = [(0, 0), (0, 0), (1, 1), (1, 1)]
        targets = ["failure", "failure", "success", "success"]

        mi_bits = mutual_information(sources, targets)

        self.assertAlmostEqual(mi_bits, 1.0, places=6)

    def test_constant_outcome_has_zero_mi(self):
        sources = [(0, 0), (0, 1), (1, 0), (1, 1)]
        targets = ["partial", "partial", "partial", "partial"]

        mi_bits = mutual_information(sources, targets)

        self.assertEqual(mi_bits, 0.0)

    def test_record_estimator_handles_smoothing_grid(self):
        records = [
            {"a1": "retrieve", "a2": "update", "y": "success"},
            {"a1": "retrieve", "a2": "verify", "y": "partial"},
            {"a1": "update", "a2": "verify", "y": "success"},
            {"a1": "error_or_noop", "a2": "verify", "y": "failure"},
        ]

        results = smoothing_sensitivity(
            records,
            source_keys=["a1", "a2"],
            target_key="y",
            alphas=[0.0, 0.5, 1.0],
        )

        self.assertEqual([result["alpha"] for result in results], [0.0, 0.5, 1.0])
        self.assertTrue(all(result["mi_bits"] >= 0.0 for result in results))

    def test_bootstrap_interval_is_ordered(self):
        records = [
            {"a1": "retrieve", "a2": "update", "y": "success"},
            {"a1": "retrieve", "a2": "verify", "y": "partial"},
            {"a1": "update", "a2": "verify", "y": "success"},
            {"a1": "error_or_noop", "a2": "verify", "y": "failure"},
            {"a1": "update", "a2": "finalize", "y": "success"},
            {"a1": "retrieve", "a2": "finalize", "y": "partial"},
        ]

        interval = bootstrap_confidence_interval(
            records,
            source_keys=["a1", "a2"],
            target_key="y",
            iterations=50,
            seed=7,
        )

        self.assertLessEqual(interval["lower_bound"], interval["point_estimate"])
        self.assertLessEqual(interval["point_estimate"], interval["upper_bound"])
        self.assertEqual(len(interval["bootstrap_estimates"]), 50)

    def test_permutation_null_distribution_returns_probability(self):
        records = [
            {"a1": "retrieve", "a2": "update", "y": "success"},
            {"a1": "retrieve", "a2": "verify", "y": "partial"},
            {"a1": "update", "a2": "verify", "y": "success"},
            {"a1": "error_or_noop", "a2": "verify", "y": "failure"},
            {"a1": "update", "a2": "finalize", "y": "success"},
            {"a1": "retrieve", "a2": "finalize", "y": "partial"},
        ]

        null_result = permutation_null_distribution(
            records,
            source_keys=["a1", "a2"],
            target_key="y",
            iterations=40,
            seed=11,
        )

        self.assertEqual(len(null_result["null_estimates"]), 40)
        self.assertGreaterEqual(null_result["p_value"], 0.0)
        self.assertLessEqual(null_result["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()