from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def iter_episode_logs(base_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(base_dir):
        if "episode.jsonl" in files:
            yield os.path.join(root, "episode.jsonl")


def load_episode_log(episode_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = []

    with open(episode_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == "episode_header":
                header = record
            elif record_type == "tool_event":
                events.append(record)

    if header is None:
        raise ValueError(f"Missing episode_header record in {episode_path}")
    return header, events


def load_optional_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_event_rows(
    *,
    base_dir: str,
    episode_path: str,
    header: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    run_dir = os.path.relpath(os.path.dirname(episode_path), base_dir)
    rows: List[Dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "run_dir": run_dir,
                "seed": header.get("seed"),
                "template_id": header.get("template_id"),
                "variant_id": header.get("variant_id"),
                "title": header.get("title"),
                "config_hash": header.get("config_hash"),
                "environment_version": header.get("environment_version"),
                "schema_version": header.get("schema_version"),
                "episode_outcome": summary.get("outcome") if summary else None,
                "sequence_id": event.get("sequence_id"),
                "agent_id": event.get("agent_id"),
                "tool_name": event.get("tool_name"),
                "valid": event.get("valid"),
                "invalid_reason": event.get("invalid_reason"),
                "invalid_detail": event.get("invalid_detail"),
                "abstraction": event.get("abstraction"),
                "progress_state": event.get("progress_state"),
                "outcome": event.get("outcome"),
                "step_count": event.get("step_count"),
                "timestamp": event.get("timestamp"),
                "arguments_json": json.dumps(event.get("arguments", {}), sort_keys=True),
                "tool_output_json": json.dumps(event.get("tool_output", {}), sort_keys=True),
                "fixed_defects_json": json.dumps(event.get("fixed_defects", []), sort_keys=True),
                "active_regressions_json": json.dumps(event.get("active_regressions", []), sort_keys=True),
                "applied_patch_ids_json": json.dumps(event.get("applied_patch_ids", []), sort_keys=True),
            }
        )
    return rows


def build_summary_row(
    *,
    base_dir: str,
    episode_path: str,
    header: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = summary or {}
    run_dir = os.path.relpath(os.path.dirname(episode_path), base_dir)
    return {
        "run_dir": run_dir,
        "seed": header.get("seed"),
        "template_id": header.get("template_id"),
        "variant_id": header.get("variant_id"),
        "title": header.get("title"),
        "config_hash": header.get("config_hash"),
        "environment_version": header.get("environment_version"),
        "schema_version": header.get("schema_version"),
        "progress_state": summary.get("progress_state"),
        "outcome": summary.get("outcome"),
        "step_count": summary.get("step_count"),
        "step_budget": summary.get("step_budget"),
        "invalid_call_rate": summary.get("invalid_call_rate"),
        "tests_passed": summary.get("tests_passed"),
        "tests_total": summary.get("tests_total"),
        "event_count": summary.get("event_count"),
        "invalid_call_counts_json": json.dumps(summary.get("invalid_call_counts", {}), sort_keys=True),
        "applied_patch_ids_json": json.dumps(summary.get("applied_patch_ids", []), sort_keys=True),
        "fixed_defects_json": json.dumps(summary.get("fixed_defects", []), sort_keys=True),
        "full_test_results_json": json.dumps(summary.get("full_test_results", []), sort_keys=True),
        "replay_spec_json": json.dumps(summary.get("replay_spec", {}), sort_keys=True),
    }


def collect_trajectory_rows(base_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    event_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    for episode_path in sorted(iter_episode_logs(base_dir)):
        try:
            header, events = load_episode_log(episode_path)
            summary_path = os.path.join(os.path.dirname(episode_path), "summary.json")
            summary = load_optional_json(summary_path)
            event_rows.extend(
                build_event_rows(
                    base_dir=base_dir,
                    episode_path=episode_path,
                    header=header,
                    events=events,
                    summary=summary,
                )
            )
            summary_rows.append(
                build_summary_row(
                    base_dir=base_dir,
                    episode_path=episode_path,
                    header=header,
                    summary=summary,
                )
            )
        except Exception as exc:
            issues.append({"episode_path": episode_path, "error": str(exc)})

    return event_rows, summary_rows, issues


def write_csv(rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_optional_parquet(rows: Sequence[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
    if not rows:
        return {"written": False, "reason": "no_rows"}

    try:
        import pandas as pd
    except ImportError:
        return {"written": False, "reason": "pandas_not_installed"}

    try:
        dataframe = pd.DataFrame(rows)
        dataframe.to_parquet(output_path, index=False)
    except Exception as exc:
        return {"written": False, "reason": str(exc)}

    return {"written": True, "path": output_path}


def export_trajectory_tables(base_dir: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    event_rows, summary_rows, issues = collect_trajectory_rows(base_dir)

    events_csv_path = os.path.join(output_dir, "trajectory_events.csv")
    summaries_csv_path = os.path.join(output_dir, "trajectory_summaries.csv")
    if event_rows:
        write_csv(event_rows, events_csv_path)
    if summary_rows:
        write_csv(summary_rows, summaries_csv_path)

    manifest = {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "episode_count": len(summary_rows),
        "event_count": len(event_rows),
        "issues": issues,
        "events_csv": events_csv_path if event_rows else None,
        "summaries_csv": summaries_csv_path if summary_rows else None,
        "events_parquet": write_optional_parquet(event_rows, os.path.join(output_dir, "trajectory_events.parquet")),
        "summaries_parquet": write_optional_parquet(summary_rows, os.path.join(output_dir, "trajectory_summaries.parquet")),
    }

    manifest_path = os.path.join(output_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export controlled tool-use trajectories to CSV and optional Parquet.")
    parser.add_argument("--base-dir", default="results", help="Root directory containing episode.jsonl outputs.")
    parser.add_argument("--output-dir", default=os.path.join("results", "exports"), help="Directory for extracted tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_trajectory_tables(args.base_dir, args.output_dir)
    print(f"Exported {manifest['episode_count']} episodes and {manifest['event_count']} events")
    print(f"Manifest written to: {os.path.join(args.output_dir, 'export_manifest.json')}")


if __name__ == "__main__":
    main()