from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def load_latest_episode(base_dir: str) -> str:
    candidates: List[Tuple[float, str]] = []
    for root, _, files in os.walk(base_dir):
        if "episode.jsonl" in files and "summary.json" in files:
            episode_path = os.path.join(root, "episode.jsonl")
            candidates.append((os.path.getmtime(episode_path), root))
    if not candidates:
        raise FileNotFoundError(f"No controlled episode outputs found under {base_dir}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_episode(run_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
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
        raise ValueError(f"Missing episode_header in {run_dir}")
    with open(os.path.join(run_dir, "summary.json"), "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return header, events, summary


def describe_episode(run_dir: str) -> None:
    header, events, summary = load_episode(run_dir)
    print(f"Run: {run_dir}")
    print(f"Fixture: {header['template_id']}/{header['variant_id']}")
    print(f"Seed: {header['seed']}")
    print(f"Outcome: {summary.get('outcome')}")
    print(f"Progress state: {summary.get('progress_state')}")
    print(f"Invalid call rate: {summary.get('invalid_call_rate')}")
    print(f"Applied patches: {summary.get('applied_patch_ids', [])}")

    per_agent: Dict[int, List[str]] = {}
    for event in events:
        agent_id = event.get("agent_id")
        if agent_id is None:
            continue
        per_agent.setdefault(agent_id, []).append(f"{event['abstraction']}:{event['tool_name']}")

    for agent_id, trace in sorted(per_agent.items()):
        print(f"Agent {agent_id}: {' > '.join(trace)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a controlled tool-use episode.")
    parser.add_argument("--run-dir", default=None, help="Specific episode run directory. Defaults to the newest run under --base-dir.")
    parser.add_argument("--base-dir", default="results", help="Root directory containing controlled episode outputs.")
    args = parser.parse_args()

    run_dir = args.run_dir or load_latest_episode(args.base_dir)
    describe_episode(run_dir)


if __name__ == "__main__":
    main()
