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


ACTION_COUNT_KEYS = (
    "retrieve_count",
    "verify_count",
    "update_count",
    "finalize_count",
    "error_or_noop_count",
)
ACTION_COUNT_BIN_CAP = 2
ACTION_PROFILE_CATEGORIES = (
    "no_actions",
    "retrieved",
    "verified",
    "updated",
    "finalized",
    "error_or_noop",
)

ABSTRACTION_TO_COUNT_KEY = {
    "retrieve": "retrieve_count",
    "verify": "verify_count",
    "update": "update_count",
    "finalize": "finalize_count",
}


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


def encode_agent_action_counts(events: Sequence[Dict[str, Any]], agent_id: int) -> Tuple[int, ...]:
    counts = {key: 0 for key in ACTION_COUNT_KEYS}
    for event in events:
        if event.get("agent_id") != agent_id:
            continue
        if not event.get("valid", True):
            counts["error_or_noop_count"] += 1
            continue
        count_key = ABSTRACTION_TO_COUNT_KEY.get(event.get("abstraction"), "error_or_noop_count")
        counts[count_key] += 1
    return tuple(min(counts[key], ACTION_COUNT_BIN_CAP) for key in ACTION_COUNT_KEYS)


def encode_agent_action_profile(events: Sequence[Dict[str, Any]], agent_id: int) -> str:
    counts = encode_agent_action_counts(events, agent_id)
    retrieve_count, verify_count, update_count, finalize_count, error_or_noop_count = counts
    if error_or_noop_count:
        return "error_or_noop"
    if finalize_count:
        return "finalized"
    if update_count:
        return "updated"
    if verify_count:
        return "verified"
    if retrieve_count:
        return "retrieved"
    return "no_actions"


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
    action_counts_by_agent = {agent_id: encode_agent_action_counts(events, agent_id) for agent_id in agent_ids}
    action_profile_by_agent = {agent_id: encode_agent_action_profile(events, agent_id) for agent_id in agent_ids}
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
        "action_count_keys": ACTION_COUNT_KEYS,
        "action_counts_by_agent": action_counts_by_agent,
        "action_profile_categories": ACTION_PROFILE_CATEGORIES,
        "action_profile_by_agent": action_profile_by_agent,
    }


def collect_episode_records(base_dir: str, *, fail_on_malformed: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    issues: List[Dict[str, str]] = []
    for run_dir in sorted(iter_episode_directories(base_dir)):
        try:
            records.append(build_episode_record(run_dir))
        except Exception as exc:
            issues.append({"run_dir": run_dir, "error": str(exc)})
            if fail_on_malformed:
                raise
    return records, {"malformed_episode_count": len(issues), "malformed_episodes": issues}


def build_mi_dataset(records: Sequence[Dict[str, Any]], *, representation: str = "coarsened_action_profile") -> List[Dict[str, Any]]:
    if representation not in {"coarsened_action_profile", "coarsened_action_counts", "full_trace"}:
        raise ValueError(f"Unsupported MI representation: {representation}")

    dataset: List[Dict[str, Any]] = []
    for record in records:
        agent_ids = record["agent_ids"]
        if len(agent_ids) < 2:
            continue
        first_id = agent_ids[0]
        second_id = agent_ids[1]
        if representation == "coarsened_action_profile":
            trace_source = record["action_profile_by_agent"]
            default_trace = "no_actions"
        elif representation == "coarsened_action_counts":
            trace_source = record["action_counts_by_agent"]
            default_trace: Any = tuple(0 for _ in ACTION_COUNT_KEYS)
        else:
            trace_source = record["trace_by_agent"]
            default_trace = "no_actions"
        dataset.append(
            {
                "baseline_name": record["baseline_name"],
                "template_id": record["template_id"],
                "variant_id": record["variant_id"],
                "seed": record["seed"],
                "representation": representation,
                "agent_0": trace_source.get(first_id, default_trace),
                "agent_1": trace_source.get(second_id, default_trace),
                "outcome": record["outcome"],
            }
        )
    return dataset


def build_one_agent_reference_dataset(
    records: Sequence[Dict[str, Any]],
    *,
    representation: str = "coarsened_action_profile",
) -> List[Dict[str, Any]]:
    if representation not in {"coarsened_action_profile", "coarsened_action_counts", "full_trace"}:
        raise ValueError(f"Unsupported MI representation: {representation}")

    dataset: List[Dict[str, Any]] = []
    for record in records:
        agent_ids = record["agent_ids"]
        if len(agent_ids) != 1:
            continue
        agent_id = agent_ids[0]
        if representation == "coarsened_action_profile":
            trace_source = record["action_profile_by_agent"]
            default_trace = "no_actions"
        elif representation == "coarsened_action_counts":
            trace_source = record["action_counts_by_agent"]
            default_trace: Any = tuple(0 for _ in ACTION_COUNT_KEYS)
        else:
            trace_source = record["trace_by_agent"]
            default_trace = "no_actions"
        dataset.append(
            {
                "baseline_name": record["baseline_name"],
                "template_id": record["template_id"],
                "variant_id": record["variant_id"],
                "seed": record["seed"],
                "representation": representation,
                "agent_0": trace_source.get(agent_id, default_trace),
                "outcome": record["outcome"],
            }
        )
    return dataset


def build_mi_report(
    mi_dataset: Sequence[Dict[str, Any]],
    *,
    representation: str,
    metric_settings: Dict[str, Any],
) -> Dict[str, Any]:
    source_keys = ["agent_0", "agent_1"]
    representation_details = {}
    if representation == "coarsened_action_profile":
        representation_details = {
            "derived_from": "coarsened_action_counts",
            "action_profile_categories": ACTION_PROFILE_CATEGORIES,
            "action_count_keys": ACTION_COUNT_KEYS,
            "count_bin_cap": ACTION_COUNT_BIN_CAP,
        }
    elif representation == "coarsened_action_counts":
        representation_details = {
            "action_count_keys": ACTION_COUNT_KEYS,
            "count_bin_cap": ACTION_COUNT_BIN_CAP,
        }
    if not mi_dataset:
        return {
            "representation": representation,
            "representation_details": representation_details,
            "source_keys": source_keys,
            "eligible_episodes": 0,
            "unique_source_state_count": 0,
            "unique_outcome_count": 0,
            "mi_bits": None,
            "status": "excluded_less_than_two_agents",
        }

    return {
        "representation": representation,
        "representation_details": representation_details,
        "source_keys": source_keys,
        "eligible_episodes": len(mi_dataset),
        "unique_source_state_count": len({tuple(record[key] for key in source_keys) for record in mi_dataset}),
        "unique_outcome_count": len({record["outcome"] for record in mi_dataset}),
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
        "status": "computed",
    }


def build_one_agent_reference_report(
    records: Sequence[Dict[str, Any]],
    *,
    representation: str,
    metric_settings: Dict[str, Any],
) -> Dict[str, Any]:
    dataset = build_one_agent_reference_dataset(records, representation=representation)
    source_keys = ["agent_0"]
    representation_details = {}
    if representation == "coarsened_action_profile":
        representation_details = {
            "derived_from": "coarsened_action_counts",
            "action_profile_categories": ACTION_PROFILE_CATEGORIES,
            "action_count_keys": ACTION_COUNT_KEYS,
            "count_bin_cap": ACTION_COUNT_BIN_CAP,
        }
    elif representation == "coarsened_action_counts":
        representation_details = {
            "action_count_keys": ACTION_COUNT_KEYS,
            "count_bin_cap": ACTION_COUNT_BIN_CAP,
        }
    report = {
        "representation": representation,
        "representation_details": representation_details,
        "source_keys": source_keys,
        "eligible_episodes": len(dataset),
        "note": "One-agent reference only; not a two-agent coordination MI estimate.",
    }
    if not dataset:
        return report
    report["one_agent_mi_bits"] = estimate_mi_from_records(
        dataset,
        source_keys=source_keys,
        target_key="outcome",
        estimator=metric_settings["estimator"],
        smoothing_alpha=0.0,
    )
    report["unique_trace_count"] = len({record["agent_0"] for record in dataset})
    return report


def summarize_coordination(base_dir: str) -> Dict[str, Any]:
    records, collection_issues = collect_episode_records(base_dir)
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
    primary_representation = "coarsened_action_profile"
    for baseline_name, baseline_records in sorted(by_baseline.items()):
        coarsened_dataset = build_mi_dataset(baseline_records, representation=primary_representation)
        action_count_dataset = build_mi_dataset(baseline_records, representation="coarsened_action_counts")
        full_trace_dataset = build_mi_dataset(baseline_records, representation="full_trace")
        coarsened_mi_report = build_mi_report(
            coarsened_dataset,
            representation=primary_representation,
            metric_settings=metric_settings,
        )
        action_count_mi_report = build_mi_report(
            action_count_dataset,
            representation="coarsened_action_counts",
            metric_settings=metric_settings,
        )
        full_trace_mi_report = build_mi_report(
            full_trace_dataset,
            representation="full_trace",
            metric_settings=metric_settings,
        )
        excluded_from_two_agent_mi = len(baseline_records) - len(coarsened_dataset)
        coarsened_mi_report["excluded_episode_count"] = excluded_from_two_agent_mi
        action_count_mi_report["excluded_episode_count"] = excluded_from_two_agent_mi
        full_trace_mi_report["excluded_episode_count"] = excluded_from_two_agent_mi
        baseline_reports[baseline_name] = {
            "episodes": len(baseline_records),
            "mi_representation": primary_representation,
            "mi_bits": coarsened_mi_report["mi_bits"],
            "two_agent_mi": coarsened_mi_report,
            "mi_representation_comparison": {
                primary_representation: coarsened_mi_report,
                "coarsened_action_counts": action_count_mi_report,
                "full_trace": full_trace_mi_report,
            },
            "one_agent_reference": build_one_agent_reference_report(
                baseline_records,
                representation=primary_representation,
                metric_settings=metric_settings,
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
        "collection_issues": collection_issues,
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
