"""Replay saved Phase 1 guesses through the offline tool-action wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tool_action_environment import (
    ACTION_NAME,
    Phase1GuessingEnvironment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RUN = Path("base-paper/results/experiment_run_20260808_192208")
FORBIDDEN_OBSERVATION_KEYS = {
    "agent_ids",
    "group_size",
    "mystery_number",
    "num_agents",
    "peer_guesses",
    "pending_guesses",
    "target",
}
EXPECTED_OBSERVATION_KEYS = {"round_num", "mode", "guess_range", "goal", "history"}
EXPECTED_HISTORY_KEYS = {"round_num", "own_guess", "feedback"}
GROUP_RESULT_KEYS = {
    "aggregate",
    "average",
    "comparison_value",
    "feedback",
    "mystery_number",
    "rounded_average",
    "target",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_commit_for(path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "log", "-1", "--format=%H", "--", path],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"No Git commit found for {path}")
    return commit


def _source_identity() -> Dict[str, Any]:
    source_path = REPOSITORY_ROOT / "base-paper" / "experiment.py"
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "repository_head": result.stdout.strip(),
        "maintained_phase1_engine_commit": _git_commit_for("base-paper/experiment.py"),
        "phase1_settings_commit": _git_commit_for("base-paper/settings.py"),
        "maintained_phase1_engine_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "reference_artifact_embeds_source_revision": False,
    }


def _normalize_guesses(guesses: Dict[Any, Any]) -> Dict[str, int]:
    return {str(agent_id): value for agent_id, value in guesses.items()}


def _observation_matches_private_history(
    observation: Dict[str, Any], expected_history: List[Dict[str, Any]]
) -> bool:
    return (
        set(observation) == EXPECTED_OBSERVATION_KEYS
        and not set(observation).intersection(FORBIDDEN_OBSERVATION_KEYS)
        and observation["history"] == expected_history
        and all(set(entry) == EXPECTED_HISTORY_KEYS for entry in observation["history"])
    )


def _reference_private_history(
    expected_rounds: Sequence[Dict[str, Any]], count: int, agent_id: int
) -> List[Dict[str, Any]]:
    return [
        {
            "round_num": previous_round["round_num"],
            "own_guess": _normalize_guesses(previous_round["guesses"])[str(agent_id)],
            "feedback": previous_round["feedback"],
        }
        for previous_round in expected_rounds[:count]
    ]


def _run_case(
    *,
    case_id: str,
    evidence_type: str,
    configuration: Dict[str, Any],
    expected_rounds: Sequence[Dict[str, Any]],
    expected_summary: Dict[str, Any],
    input_trajectory_paths: Sequence[str],
    scenario: str,
) -> Dict[str, Any]:
    agent_ids = [agent["agent_id"] for agent in configuration["agents"]]
    environment = Phase1GuessingEnvironment(
        mode=configuration["mode"],
        target=configuration["mystery_number"],
        target_range=tuple(configuration["mystery_range"]),
        guess_range=tuple(configuration["guess_range"]),
        agent_ids=agent_ids,
        max_rounds=configuration["max_rounds"],
    )

    mismatches: List[Dict[str, Any]] = []
    round_comparisons: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    feedback_withheld = True
    observation_restrictions_preserved = True
    observation_checks = 0

    for expected_round in expected_rounds:
        expected_guesses = _normalize_guesses(expected_round["guesses"])
        visible_feedback_count = len(environment._round_results)

        for agent_id in agent_ids:
            observation = environment.observe(agent_id)
            observation_checks += 1
            prior_history = _reference_private_history(
                expected_rounds, visible_feedback_count, agent_id
            )
            if not _observation_matches_private_history(observation, prior_history):
                observation_restrictions_preserved = False
                mismatches.append(
                    {
                        "round_num": expected_round["round_num"],
                        "field": "agent_observation_restrictions",
                        "agent_id": agent_id,
                    }
                )

        for submission_index, agent_id in enumerate(agent_ids):
            event = environment.step(
                agent_id,
                ACTION_NAME,
                {"value": expected_guesses[str(agent_id)]},
            )
            events.append(event.to_dict())
            if not event.valid:
                mismatches.append(
                    {
                        "round_num": expected_round["round_num"],
                        "field": "accepted_action",
                        "agent_id": agent_id,
                        "invalid_reason": event.invalid_reason,
                    }
                )
            if set(event.tool_output).intersection(GROUP_RESULT_KEYS):
                feedback_withheld = False
                mismatches.append(
                    {
                        "round_num": expected_round["round_num"],
                        "field": "group_result_in_tool_output",
                        "agent_id": agent_id,
                    }
                )
            if submission_index < len(agent_ids) - 1:
                round_completed = len(environment._round_results) > visible_feedback_count
                if round_completed:
                    feedback_withheld = False
                for observer_id in agent_ids:
                    partial_observation = environment.observe(observer_id)
                    prior_history = _reference_private_history(
                        expected_rounds, visible_feedback_count, observer_id
                    )
                    if not _observation_matches_private_history(partial_observation, prior_history):
                        feedback_withheld = False
                        observation_restrictions_preserved = False
                        mismatches.append(
                            {
                                "round_num": expected_round["round_num"],
                                "field": "premature_feedback",
                                "agent_id": observer_id,
                            }
                        )

        if len(environment._round_results) != visible_feedback_count + 1:
            mismatches.append(
                {
                    "round_num": expected_round["round_num"],
                    "field": "round_resolution_count",
                }
            )
            continue

        replayed_round = environment._round_results[-1]
        replayed_guesses = _normalize_guesses(replayed_round["guesses"])
        checks = {
            "individual_guesses_and_actor_ids": replayed_guesses == expected_guesses,
            "average_field": replayed_round["average"] == expected_round["average"],
            "comparison_value": replayed_round["comparison_value"]
            == expected_round["rounded_average"],
            "rounded_average_field": replayed_round["rounded_average"]
            == expected_round["rounded_average"],
            "feedback": replayed_round["feedback"] == expected_round["feedback"],
        }
        round_mismatches = [field for field, passed in checks.items() if not passed]
        for field in round_mismatches:
            mismatches.append({"round_num": expected_round["round_num"], "field": field})
        round_comparisons.append(
            {
                "round_num": expected_round["round_num"],
                "reference": {
                    "guesses": expected_guesses,
                    "average": expected_round["average"],
                    "rounded_average": expected_round["rounded_average"],
                    "feedback": expected_round["feedback"],
                },
                "replay": {
                    "guesses": replayed_guesses,
                    "average": replayed_round["average"],
                    "rounded_average": replayed_round["rounded_average"],
                    "comparison_value": replayed_round["comparison_value"],
                    "feedback": replayed_round["feedback"],
                },
                "checks": checks,
                "mismatches": round_mismatches,
            }
        )

        if environment._execution_status == "completed":
            break

    expected_solved = expected_summary["solved"]
    expected_outcome = "solved" if expected_solved else "unsolved"
    actual_round_count = len(environment._round_results)
    actual_stopping_round = (
        environment._round_results[-1]["round_num"]
        if environment._round_results
        else None
    )
    expected_round_count = expected_summary["total_rounds"]
    actual_outcome = environment._outcome
    terminal_checks = {
        "round_count": actual_round_count == expected_round_count,
        "reference_round_artifact_count": actual_round_count == len(expected_rounds),
        "stopping_round": actual_stopping_round == expected_round_count,
        "native_outcome": actual_outcome == expected_outcome,
        "execution_completed": environment._execution_status == "completed"
        and expected_summary.get("completed_successfully") is True,
    }
    for field, passed in terminal_checks.items():
        if not passed:
            mismatches.append({"field": f"terminal_{field}"})

    return {
        "case_id": case_id,
        "evidence_type": evidence_type,
        "scenario": scenario,
        "input_trajectory_paths": list(input_trajectory_paths),
        "configuration": configuration,
        "rounds_checked": len(round_comparisons),
        "actions_checked": len(events),
        "round_comparisons": round_comparisons,
        "events": events,
        "protocol_checks": {
            "feedback_withheld_before_all_round_submissions": feedback_withheld,
            "agent_facing_private_observations_preserved": observation_restrictions_preserved,
            "agent_observations_checked": observation_checks,
        },
        "terminal": {
            "reference_round_count": expected_round_count,
            "replay_round_count": actual_round_count,
            "reference_outcome": expected_outcome,
            "replay_outcome": actual_outcome,
            "execution_status": environment._execution_status,
            "checks": terminal_checks,
        },
        "mismatches": mismatches,
        "passed": not mismatches
        and feedback_withheld
        and observation_restrictions_preserved
        and all(terminal_checks.values()),
    }


def _historical_case() -> Dict[str, Any]:
    run_dir = REPOSITORY_ROOT / HISTORICAL_RUN
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    round_paths = sorted(run_dir.glob("round_*.json"))
    expected_rounds = [_read_json(path) for path in round_paths]
    input_paths = [
        (HISTORICAL_RUN / "config.json").as_posix(),
        (HISTORICAL_RUN / "summary.json").as_posix(),
        *[(HISTORICAL_RUN / path.name).as_posix() for path in round_paths],
    ]
    return _run_case(
        case_id="phase1_saved_sum_unsolved_20260808",
        evidence_type="historical",
        configuration=config,
        expected_rounds=expected_rounds,
        expected_summary=summary,
        input_trajectory_paths=input_paths,
        scenario="Saved five-agent sum-mode trajectory; stopped unsolved at the 20-round limit.",
    )


def _synthetic_mean_case() -> Dict[str, Any]:
    configuration = {
        "mode": "mean",
        "mystery_number": 2,
        "mystery_range": [0, 10],
        "guess_range": [0, 10],
        "max_rounds": 3,
        "agents": [{"agent_id": 0}, {"agent_id": 1}],
    }
    expected_rounds = [
        {
            "round_num": 1,
            "guesses": {"0": 0, "1": 1},
            "average": 0.5,
            "rounded_average": 0,
            "feedback": "too LOW ",
        },
        {
            "round_num": 2,
            "guesses": {"0": 1, "1": 2},
            "average": 1.5,
            "rounded_average": 2,
            "feedback": "CORRECT! ",
        },
    ]
    return _run_case(
        case_id="synthetic_mean_half_rounding_early_stop",
        evidence_type="synthetic_regression",
        configuration=configuration,
        expected_rounds=expected_rounds,
        expected_summary={
            "total_rounds": 2,
            "solved": True,
            "completed_successfully": True,
        },
        input_trajectory_paths=[],
        scenario=(
            "Hand-specified regression case: Python ties-to-even mean rounding at 0.5 "
            "and 1.5, followed by native solved early stopping."
        ),
    )


def run_replay(output_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    execution_date = date.today().isoformat()
    historical = _historical_case()
    synthetic = _synthetic_mean_case()
    report = {
        "bridge_version": "phase1-tool-action-bridge-v1",
        "replay_execution_date": execution_date,
        "reference_source_identity": _source_identity(),
        "historical_reference_limitations": (
            "The saved config does not embed a source revision. The maintained source "
            "commit and file hash are identified separately; exact local run-time source "
            "cannot be proven from this artifact."
        ),
        "legacy_field_interpretation": {
            "average": "Group sum in sum mode; unrounded arithmetic mean in mean mode.",
            "rounded_average": (
                "Value compared with the target: the sum in sum mode; Python round(mean) "
                "in mean mode."
            ),
        },
        "case_count": 2,
        "trajectory_count": 1,
        "rounds_checked": historical["rounds_checked"] + synthetic["rounds_checked"],
        "actions_checked": historical["actions_checked"] + synthetic["actions_checked"],
        "historical_trajectory_count": 1,
        "synthetic_case_count": 1,
        "cases": [historical, synthetic],
        "mismatches": [
            {"case_id": case["case_id"], **mismatch}
            for case in (historical, synthetic)
            for mismatch in case["mismatches"]
        ],
        "passed": historical["passed"] and synthetic["passed"],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "replay_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report_path, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "base-paper"
        / "results"
        / f"phase1_tool_bridge_{date.today():%Y%m%d}",
        help="Directory for the bridge-only replay report (default: dated base-paper/results folder).",
    )
    args = parser.parse_args(argv)
    report_path, report = run_replay(args.output_dir.resolve())
    print(
        json.dumps(
            {
                "report": str(report_path),
                "passed": report["passed"],
                "case_count": report["case_count"],
                "trajectory_count": report["trajectory_count"],
                "rounds_checked": report["rounds_checked"],
                "mismatch_count": len(report["mismatches"]),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
