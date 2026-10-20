"""Offline tool-action interface for the maintained Phase 1 guessing game."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


ACTION_NAME = "submit_guess"
ACTION_CATEGORY = "numerical_contribution"

NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
OUTCOME_SOLVED = "solved"
OUTCOME_UNSOLVED = "unsolved"


@dataclass
class GuessActionEvent:
    record_type: str
    sequence_id: int
    round_num: int
    agent_id: Optional[int]
    tool_name: str
    arguments: Dict[str, Any]
    valid: bool
    invalid_reason: Optional[str]
    invalid_detail: Optional[str]
    action_category: str
    abstraction: str
    progress_state: str
    outcome: Optional[str]
    tool_output: Dict[str, Any]
    step_count: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase1GuessingEnvironment:
    """Collect one private numerical contribution per agent before feedback."""

    def __init__(
        self,
        *,
        mode: str,
        target: int,
        target_range: Tuple[int, int],
        guess_range: Tuple[int, int],
        agent_ids: Sequence[int],
        max_rounds: int,
    ) -> None:
        if mode not in {"sum", "mean"}:
            raise ValueError("mode must be 'sum' or 'mean'")
        if not self._valid_range(target_range) or not self._valid_range(guess_range):
            raise ValueError("target_range and guess_range must be ordered integer pairs")
        if not target_range[0] <= target <= target_range[1]:
            raise ValueError("target must be within target_range")
        if not agent_ids or any(type(agent_id) is not int for agent_id in agent_ids):
            raise ValueError("agent_ids must contain at least one integer ID")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_ids must be unique")
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")

        self._mode = mode
        self._target = target
        self._target_range = target_range
        self._guess_range = guess_range
        self._agent_ids = tuple(agent_ids)
        self._max_rounds = max_rounds
        self._current_round = 1
        self._progress_state = NOT_STARTED
        self._outcome: Optional[str] = None
        self._execution_status = "in_progress"
        self._step_count = 0
        self._event_sequence = 0
        self._round_results: List[Dict[str, Any]] = []
        self._pending_guesses: Dict[int, int] = {}
        self._private_history: Dict[int, List[int]] = {
            agent_id: [] for agent_id in self._agent_ids
        }
        self._last_observed_round_count: Dict[int, int] = {
            agent_id: 0 for agent_id in self._agent_ids
        }
        self._shared_feedback: List[str] = []

    @staticmethod
    def _valid_range(value: Tuple[int, int]) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and all(type(bound) is int for bound in value)
            and value[0] <= value[1]
        )

    def list_valid_actions(self) -> List[Dict[str, Any]]:
        if self._execution_status != "in_progress":
            return []
        return [
            {
                "tool_name": ACTION_NAME,
                "arguments_schema": {
                    "value": {
                        "type": "integer",
                        "minimum": self._guess_range[0],
                        "maximum": self._guess_range[1],
                    }
                },
            }
        ]

    def observe(self, agent_id: int) -> Dict[str, Any]:
        if type(agent_id) is not int or agent_id not in self._private_history:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        self._last_observed_round_count[agent_id] = len(self._round_results)
        return {
            "round_num": self._current_round,
            "mode": self._mode,
            "guess_range": list(self._guess_range),
            "goal": "make the group sum equal the target"
            if self._mode == "sum"
            else "make the rounded group mean equal the target",
            "history": [
                {
                    "round_num": index + 1,
                    "own_guess": guess,
                    "feedback": self._shared_feedback[index],
                }
                for index, guess in enumerate(self._private_history[agent_id])
            ],
        }

    def step(
        self, agent_id: int, tool_name: str, arguments: Dict[str, Any]
    ) -> GuessActionEvent:
        action_category = ACTION_CATEGORY if tool_name == ACTION_NAME else "unsupported"
        action_round = self._current_round
        invalid_reason: Optional[str] = None
        invalid_detail: Optional[str] = None

        if tool_name != ACTION_NAME:
            invalid_reason = "schema_invalid"
            invalid_detail = "unknown_tool_name"
        elif self._execution_status != "in_progress":
            invalid_reason = "state_invalid"
            invalid_detail = "game_already_completed"
        elif type(agent_id) is not int or agent_id not in self._private_history:
            invalid_reason = "reference_invalid"
            invalid_detail = "unknown_agent_id"
        elif self._last_observed_round_count[agent_id] < len(self._round_results):
            invalid_reason = "state_invalid"
            invalid_detail = "feedback_not_observed"
        elif not isinstance(arguments, dict) or set(arguments) != {"value"}:
            invalid_reason = "schema_invalid"
            invalid_detail = "expected_only_value_argument"
        elif type(arguments["value"]) is not int:
            invalid_reason = "schema_invalid"
            invalid_detail = "value_must_be_integer"
        elif not self._guess_range[0] <= arguments["value"] <= self._guess_range[1]:
            invalid_reason = "schema_invalid"
            invalid_detail = "value_out_of_range"
        elif agent_id in self._pending_guesses:
            invalid_reason = "state_invalid"
            invalid_detail = "agent_already_submitted_this_round"

        self._step_count += 1
        if invalid_reason is None:
            if self._progress_state == NOT_STARTED:
                self._progress_state = IN_PROGRESS
            self._pending_guesses[agent_id] = arguments["value"]
            if len(self._pending_guesses) == len(self._agent_ids):
                self._resolve_round()

        event = GuessActionEvent(
            record_type="tool_event",
            sequence_id=self._event_sequence + 1,
            round_num=action_round,
            agent_id=agent_id if type(agent_id) is int else None,
            tool_name=tool_name,
            arguments=copy.deepcopy(arguments),
            valid=invalid_reason is None,
            invalid_reason=invalid_reason,
            invalid_detail=invalid_detail,
            action_category=action_category,
            abstraction="contribute" if tool_name == ACTION_NAME else "error_or_noop",
            progress_state=self._progress_state,
            outcome=self._outcome,
            tool_output={"accepted": True} if invalid_reason is None else {"accepted": False},
            step_count=self._step_count,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._event_sequence += 1
        return event

    def _resolve_round(self) -> None:
        guesses = dict(self._pending_guesses)
        if self._mode == "sum":
            average = sum(guesses.values())
            comparison_value = average
        else:
            average = sum(guesses.values()) / len(guesses)
            comparison_value = round(average)

        if comparison_value == self._target:
            feedback = "CORRECT! "
        elif comparison_value > self._target:
            feedback = "too HIGH "
        else:
            feedback = "too LOW "

        result = {
            "round_num": self._current_round,
            "guesses": guesses,
            "average": average,
            "rounded_average": comparison_value,
            "comparison_value": comparison_value,
            "feedback": feedback,
            "mystery_number": self._target,
            "mode": self._mode,
        }
        self._round_results.append(result)
        for agent_id, guess in guesses.items():
            self._private_history[agent_id].append(guess)
        self._shared_feedback.append(feedback)
        self._pending_guesses.clear()

        if feedback.startswith("CORRECT"):
            self._outcome = OUTCOME_SOLVED
            self._execution_status = "completed"
            self._progress_state = COMPLETED
        elif self._current_round >= self._max_rounds:
            self._outcome = OUTCOME_UNSOLVED
            self._execution_status = "completed"
            self._progress_state = COMPLETED
        else:
            self._current_round += 1
