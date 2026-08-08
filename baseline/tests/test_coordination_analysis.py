import json
import os
import tempfile
import unittest

from coordination_analysis import build_episode_record, build_mi_dataset, summarize_coordination


class CoordinationAnalysisTests(unittest.TestCase):
    def _write_episode(self, root_dir: str, baseline_name: str, outcome: str) -> str:
        run_dir = os.path.join(root_dir, f"run_{baseline_name}_{outcome}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "episode.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "episode_header",
                "seed": 1,
                "template_id": "math_ops",
                "variant_id": "sum_bounds",
            }) + "\n")
            handle.write(json.dumps({
                "record_type": "tool_event",
                "agent_id": 0,
                "abstraction": "retrieve",
                "tool_name": "retrieve_file",
                "valid": True,
            }) + "\n")
            handle.write(json.dumps({
                "record_type": "tool_event",
                "agent_id": 1,
                "abstraction": "update",
                "tool_name": "apply_patch",
                "valid": True,
            }) + "\n")
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

        dataset = build_mi_dataset(records)

        self.assertEqual(dataset[0]["agent_0"], "retrieve")
        self.assertEqual(dataset[0]["agent_1"], "update")

    def test_summarize_coordination_groups_by_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_episode(temp_dir, "single_agent", "success")
            self._write_episode(temp_dir, "random_policy", "failure")

            report = summarize_coordination(temp_dir)

            self.assertEqual(report["episodes"], 2)
            self.assertIn("single_agent", report["baselines"])
            self.assertIn("random_policy", report["baselines"])


if __name__ == "__main__":
    unittest.main()