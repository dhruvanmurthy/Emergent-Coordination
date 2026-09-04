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

    def test_independent_traces_have_low_mi(self):
        sources = []
        targets = []
        for source in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            sources.extend([source, source])
            targets.extend(["success", "failure"])

        mi_bits = mutual_information(sources, targets)

        self.assertLess(mi_bits, 0.01)

    def test_shuffled_outcomes_reduce_mi(self):
        aligned_sources = ["retrieve_heavy"] * 8 + ["update_heavy"] * 8
        aligned_targets = ["partial"] * 8 + ["success"] * 8
        shuffled_targets = ["partial", "success"] * 8

        aligned_mi = mutual_information(aligned_sources, aligned_targets)
        shuffled_mi = mutual_information(aligned_sources, shuffled_targets)

        self.assertGreater(aligned_mi, shuffled_mi + 0.5)

    def test_duplicated_agents_do_not_increase_mi(self):
        agent_trace = ["retrieve_heavy"] * 6 + ["update_heavy"] * 6
        duplicated_agent_trace = [(trace, trace) for trace in agent_trace]
        outcomes = ["partial"] * 6 + ["success"] * 6

        single_agent_mi = mutual_information(agent_trace, outcomes)
        duplicated_agent_mi = mutual_information(duplicated_agent_trace, outcomes)

        self.assertAlmostEqual(single_agent_mi, duplicated_agent_mi, places=6)

    def test_complementary_scripted_agents_have_joint_mi(self):
        agent_0 = [0, 0, 1, 1] * 4
        agent_1 = [0, 1, 0, 1] * 4
        joint_trace = list(zip(agent_0, agent_1))
        outcomes = ["success" if first != second else "failure" for first, second in joint_trace]

        agent_0_mi = mutual_information(agent_0, outcomes)
        agent_1_mi = mutual_information(agent_1, outcomes)
        joint_mi = mutual_information(joint_trace, outcomes)

        self.assertLess(agent_0_mi, 0.01)
        self.assertLess(agent_1_mi, 0.01)
        self.assertGreater(joint_mi, 0.5)


if __name__ == "__main__":
    unittest.main()