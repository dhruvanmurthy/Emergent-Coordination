from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from coordination_metrics import (
    bootstrap_confidence_interval,
    estimate_mi_from_records,
    permutation_null_distribution,
    smoothing_sensitivity,
)
from settings import get_required


def iter_episode_directories(base_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(base_dir):
        if "episode.jsonl" in files and "summary.json" in files:
            yield root


def load_episode_records(run_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    header: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = []
    with open(os.path.join(run_dir, "episode.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "episode_header":
                header = record
            elif record.get("record_type") == "tool_event":
                events.append(record)

    if header is None:
        raise ValueError(f"Missing episode header in {run_dir}")

    with open(os.path.join(run_dir, "summary.json"), "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return header, events, summary


def encode_agent_trace(events: Sequence[Dict[str, Any]], agent_id: int) -> str:
    abstractions = [event["abstraction"] for event in events if event.get("agent_id") == agent_id]
    return ">".join(abstractions) if abstractions else "no_actions"


def invalid_recovery_rate(events: Sequence[Dict[str, Any]]) -> float:
    invalid_positions = [index for index, event in enumerate(events) if not event.get("valid", True)]
    if not invalid_positions:
        return 0.0
    recoveries = 0
    for index in invalid_positions:
        if index + 1 < len(events) and events[index + 1].get("valid", False):
            recoveries += 1
    return recoveries / len(invalid_positions)


def build_episode_record(run_dir: str) -> Dict[str, Any]:
    header, events, summary = load_episode_records(run_dir)
    agent_ids = sorted({event.get("agent_id") for event in events if event.get("agent_id") is not None})
    trace_by_agent = {agent_id: encode_agent_trace(events, agent_id) for agent_id in agent_ids}
    return {
        "run_dir": run_dir,
        "seed": header.get("seed"),
        "template_id": header.get("template_id"),
        "variant_id": header.get("variant_id"),
        "baseline_name": summary.get("baseline_name", "unknown"),
        "outcome": summary.get("outcome"),
        "invalid_call_rate": summary.get("invalid_call_rate", 0.0),
        "calls_to_completion": summary.get("calls_to_completion", summary.get("event_count", 0)),
        "patch_acceptance_ratio": summary.get("patch_acceptance_ratio", 0.0),
        "test_run_efficiency": summary.get("test_run_efficiency", 0.0),
        "invalid_recovery_rate": invalid_recovery_rate(events),
        "agent_ids": agent_ids,
        "trace_by_agent": trace_by_agent,
    }


def collect_episode_records(base_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for run_dir in sorted(iter_episode_directories(base_dir)):
        try:
            records.append(build_episode_record(run_dir))
        except Exception:
            continue
    return records


def build_mi_dataset(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    for record in records:
        agent_ids = record["agent_ids"]
        first_id = agent_ids[0] if agent_ids else 0
        second_id = agent_ids[1] if len(agent_ids) > 1 else first_id
        dataset.append(
            {
                "baseline_name": record["baseline_name"],
                "template_id": record["template_id"],
                "variant_id": record["variant_id"],
                "seed": record["seed"],
                "agent_0": record["trace_by_agent"].get(first_id, "no_actions"),
                "agent_1": record["trace_by_agent"].get(second_id, "no_actions"),
                "outcome": record["outcome"],
            }
        )
    return dataset


def summarize_coordination(base_dir: str) -> Dict[str, Any]:
    records = collect_episode_records(base_dir)
    metric_settings = {
        "estimator": get_required("coordination_metrics", "estimator"),
        "smoothing_alphas": get_required("coordination_metrics", "smoothing_alphas"),
        "bootstrap_iterations": get_required("coordination_metrics", "bootstrap_iterations"),
        "permutation_iterations": get_required("coordination_metrics", "permutation_iterations"),
        "confidence_level": get_required("coordination_metrics", "confidence_level"),
    }

    by_baseline: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_baseline[record["baseline_name"]].append(record)

    baseline_reports: Dict[str, Any] = {}
    for baseline_name, baseline_records in sorted(by_baseline.items()):
        mi_dataset = build_mi_dataset(baseline_records)
        if not mi_dataset:
            continue
        source_keys = ["agent_0", "agent_1"]
        baseline_reports[baseline_name] = {
            "episodes": len(baseline_records),
            "mi_bits": estimate_mi_from_records(
                mi_dataset,
                source_keys=source_keys,
                target_key="outcome",
                estimator=metric_settings["estimator"],
                smoothing_alpha=0.0,
            ),
            "smoothing_sensitivity": smoothing_sensitivity(
                mi_dataset,
                source_keys=source_keys,
                target_key="outcome",
                alphas=metric_settings["smoothing_alphas"],
                estimator=metric_settings["estimator"],
            ),
            "bootstrap": bootstrap_confidence_interval(
                mi_dataset,
                source_keys=source_keys,
                target_key="outcome",
                estimator=metric_settings["estimator"],
                iterations=metric_settings["bootstrap_iterations"],
                confidence_level=metric_settings["confidence_level"],
                seed=17,
            ),
            "permutation_null": permutation_null_distribution(
                mi_dataset,
                source_keys=source_keys,
                target_key="outcome",
                estimator=metric_settings["estimator"],
                iterations=metric_settings["permutation_iterations"],
                seed=23,
            ),
            "mean_invalid_call_rate": sum(record["invalid_call_rate"] for record in baseline_records) / len(baseline_records),
            "mean_calls_to_completion": sum(record["calls_to_completion"] for record in baseline_records) / len(baseline_records),
            "mean_patch_acceptance_ratio": sum(record["patch_acceptance_ratio"] for record in baseline_records) / len(baseline_records),
            "mean_invalid_recovery_rate": sum(record["invalid_recovery_rate"] for record in baseline_records) / len(baseline_records),
            "mean_test_run_efficiency": sum(record["test_run_efficiency"] for record in baseline_records) / len(baseline_records),
        }

    return {
        "base_dir": base_dir,
        "episodes": len(records),
        "baselines": baseline_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize coordination metrics from controlled tool-use runs.")
    parser.add_argument("--base-dir", default="results", help="Root directory containing episode outputs.")
    parser.add_argument("--output-path", default=os.path.join("results", "coordination_report.json"), help="Path for the coordination summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_coordination(args.base_dir)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote coordination report to {args.output_path}")


if __name__ == "__main__":
    main()
