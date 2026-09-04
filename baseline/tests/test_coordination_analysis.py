import json
import os
import tempfile
import unittest

from coordination_analysis import (
    ACTION_COUNT_BIN_CAP,
    ACTION_COUNT_KEYS,
    ACTION_PROFILE_CATEGORIES,
    build_episode_record,
    build_mi_dataset,
    collect_episode_records,
    encode_agent_action_counts,
    encode_agent_action_profile,
    summarize_coordination,
)


class CoordinationAnalysisTests(unittest.TestCase):
    def _write_episode(self, root_dir: str, baseline_name: str, outcome: str, events=None, run_name=None) -> str:
        run_dir = os.path.join(root_dir, run_name or f"run_{baseline_name}_{outcome}")
        os.makedirs(run_dir, exist_ok=True)
        if events is None:
            events = [
                {
                    "record_type": "tool_event",
                    "agent_id": 0,
                    "abstraction": "retrieve",
                    "tool_name": "retrieve_file",
                    "valid": True,
                },
                {
                    "record_type": "tool_event",
                    "agent_id": 1,
                    "abstraction": "update",
                    "tool_name": "apply_patch",
                    "valid": True,
                },
            ]
        with open(os.path.join(run_dir, "episode.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "episode_header",
                "seed": 1,
                "template_id": "math_ops",
                "variant_id": "sum_bounds",
            }) + "\n")
            for event in events:
                handle.write(json.dumps(event) + "\n")
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "baseline_name": baseline_name,
                "outcome": outcome,
                "invalid_call_rate": 0.0,
                "calls_to_completion": 2,
                "patch_acceptance_ratio": 1.0,
                "test_run_efficiency": 0.5,
            }, handle)
        return run_dir

    def test_build_episode_record_extracts_agent_traces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self._write_episode(temp_dir, "single_agent", "success")
            record = build_episode_record(run_dir)

            self.assertEqual(record["baseline_name"], "single_agent")
            self.assertEqual(record["trace_by_agent"][0], "retrieve")
            self.assertEqual(record["trace_by_agent"][1], "update")

    def test_coarsened_trace_encoding_counts_stable_categories(self):
        events = [
            {"agent_id": 0, "abstraction": "retrieve", "valid": True},
            {"agent_id": 0, "abstraction": "verify", "valid": True},
            {"agent_id": 0, "abstraction": "update", "valid": True},
            {"agent_id": 0, "abstraction": "finalize", "valid": True},
            {"agent_id": 0, "abstraction": "retrieve", "valid": False},
            {"agent_id": 0, "abstraction": "unexpected", "valid": True},
            {"agent_id": 1, "abstraction": "retrieve", "valid": True},
        ]

        encoded = encode_agent_action_counts(events, agent_id=0)

        self.assertEqual(ACTION_COUNT_KEYS, (
            "retrieve_count",
            "verify_count",
            "update_count",
            "finalize_count",
            "error_or_noop_count",
        ))
        self.assertEqual(ACTION_COUNT_BIN_CAP, 2)
        self.assertEqual(encoded, (1, 1, 1, 1, 2))

    def test_coarsened_trace_encoding_caps_repeated_actions(self):
        events = [
            {"agent_id": 0, "abstraction": "retrieve", "valid": True},
            {"agent_id": 0, "abstraction": "retrieve", "valid": True},
            {"agent_id": 0, "abstraction": "retrieve", "valid": True},
        ]

        encoded = encode_agent_action_counts(events, agent_id=0)

        self.assertEqual(encoded, (2, 0, 0, 0, 0))

    def test_action_profile_uses_stable_categories(self):
        events = [
            {"agent_id": 0, "abstraction": "retrieve", "valid": True},
            {"agent_id": 0, "abstraction": "verify", "valid": True},
            {"agent_id": 0, "abstraction": "update", "valid": True},
        ]

        profile = encode_agent_action_profile(events, agent_id=0)

        self.assertEqual(ACTION_PROFILE_CATEGORIES, (
            "no_actions",
            "retrieved",
            "verified",
            "updated",
            "finalized",
            "error_or_noop",
        ))
        self.assertEqual(profile, "updated")

    def test_build_mi_dataset_normalizes_agent_columns(self):
        records = [{
            "baseline_name": "single_agent",
            "template_id": "math_ops",
            "variant_id": "sum_bounds",
            "seed": 1,
            "outcome": "success",
            "agent_ids": [0, 1],
            "trace_by_agent": {0: "retrieve", 1: "update"},
        }]

        dataset = build_mi_dataset(records, representation="full_trace")

        self.assertEqual(dataset[0]["agent_0"], "retrieve")
        self.assertEqual(dataset[0]["agent_1"], "update")

    def test_build_mi_dataset_uses_coarsened_counts_by_default(self):
        records = [{
            "baseline_name": "independent_multi_agent",
            "template_id": "math_ops",
            "variant_id": "sum_bounds",
            "seed": 1,
            "outcome": "success",
            "agent_ids": [0, 1],
            "trace_by_agent": {0: "retrieve>verify", 1: "update>finalize"},
            "action_counts_by_agent": {0: (1, 1, 0, 0, 0), 1: (0, 0, 1, 1, 0)},
            "action_profile_by_agent": {0: "verified", 1: "finalized"},
        }]

        dataset = build_mi_dataset(records)

        self.assertEqual(dataset[0]["representation"], "coarsened_action_profile")
        self.assertEqual(dataset[0]["agent_0"], "verified")
        self.assertEqual(dataset[0]["agent_1"], "finalized")

        count_dataset = build_mi_dataset(records, representation="coarsened_action_counts")
        self.assertEqual(count_dataset[0]["agent_0"], (1, 1, 0, 0, 0))
        self.assertEqual(count_dataset[0]["agent_1"], (0, 0, 1, 1, 0))

    def test_single_agent_is_excluded_from_two_agent_mi(self):
        events = [
            {"record_type": "tool_event", "agent_id": 0, "abstraction": "retrieve", "valid": True},
            {"record_type": "tool_event", "agent_id": 0, "abstraction": "update", "valid": True},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_episode(temp_dir, "single_agent", "success", events=events)

            report = summarize_coordination(temp_dir)

            baseline_report = report["baselines"]["single_agent"]
            self.assertIsNone(baseline_report["mi_bits"])
            self.assertEqual(baseline_report["two_agent_mi"]["status"], "excluded_less_than_two_agents")
            self.assertEqual(baseline_report["two_agent_mi"]["excluded_episode_count"], 1)
            self.assertEqual(baseline_report["one_agent_reference"]["eligible_episodes"], 1)

    def test_summarize_coordination_groups_by_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_episode(temp_dir, "single_agent", "success")
            self._write_episode(temp_dir, "random_policy", "failure")

            report = summarize_coordination(temp_dir)

            self.assertEqual(report["episodes"], 2)
            self.assertIn("single_agent", report["baselines"])
            self.assertIn("random_policy", report["baselines"])

    def test_mi_report_groups_by_representation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_episode(temp_dir, "random_policy", "success", run_name="run_random_policy_success_1")
            self._write_episode(temp_dir, "random_policy", "failure", run_name="run_random_policy_failure_1")

            report = summarize_coordination(temp_dir)

            baseline_report = report["baselines"]["random_policy"]
            self.assertEqual(baseline_report["mi_representation"], "coarsened_action_profile")
            self.assertEqual(baseline_report["two_agent_mi"]["representation"], "coarsened_action_profile")
            self.assertEqual(baseline_report["two_agent_mi"]["representation_details"]["count_bin_cap"], 2)
            self.assertIn("unique_source_state_count", baseline_report["two_agent_mi"])
            self.assertIn("coarsened_action_profile", baseline_report["mi_representation_comparison"])
            self.assertIn("coarsened_action_counts", baseline_report["mi_representation_comparison"])
            self.assertIn("full_trace", baseline_report["mi_representation_comparison"])

    def test_collect_episode_records_exposes_malformed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            malformed_dir = os.path.join(temp_dir, "malformed_run")
            os.makedirs(malformed_dir, exist_ok=True)
            with open(os.path.join(malformed_dir, "episode.jsonl"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"record_type": "tool_event", "agent_id": 0}) + "\n")
            with open(os.path.join(malformed_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump({"baseline_name": "random_policy", "outcome": "failure"}, handle)

            records, issues = collect_episode_records(temp_dir)

            self.assertEqual(records, [])
            self.assertEqual(issues["malformed_episode_count"], 1)
            self.assertIn("Missing episode header", issues["malformed_episodes"][0]["error"])


if __name__ == "__main__":
    unittest.main()