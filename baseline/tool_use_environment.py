from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ENVIRONMENT_VERSION = "synthetic-bugfix-v1"
SCHEMA_VERSION = "tool-use-episode-v1"

NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
COMPLETED_SUCCESS = "completed_success"
COMPLETED_PARTIAL = "completed_partial"
COMPLETED_FAILURE = "completed_failure"

OUTCOME_SUCCESS = "success"
OUTCOME_PARTIAL = "partial"
OUTCOME_FAILURE = "failure"

INVALID_SCHEMA = "schema_invalid"
INVALID_REFERENCE = "reference_invalid"
INVALID_STATE = "state_invalid"
INVALID_NOOP = "semantic_noop"
INVALID_BUDGET = "budget_invalid"

VALID_TOOL_NAMES = {
    "retrieve_file",
    "search_symbol",
    "run_tests",
    "apply_patch",
    "finalize_ticket",
}

ABSTRACTION_BY_TOOL = {
    "retrieve_file": "retrieve",
    "search_symbol": "retrieve",
    "run_tests": "verify",
    "apply_patch": "update",
    "finalize_ticket": "finalize",
}


@dataclass(frozen=True)
class PatchSpec:
    patch_id: str
    patch: str
    file_path: str
    old_text: str
    new_text: str
    fixes_defects: Tuple[str, ...]
    introduces_regressions: Tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class TestTargetSpec:
    name: str
    checks_defects: Tuple[str, ...]
    regression_guards: Tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class BugfixTaskVariant:
    template_id: str
    variant_id: str
    title: str
    description: str
    files: Dict[str, str]
    symbols: Dict[str, List[Dict[str, Any]]]
    patches: Tuple[PatchSpec, ...]
    test_targets: Tuple[TestTargetSpec, ...]
    required_defects: Tuple[str, ...]
    max_regression_failures: int = 0


@dataclass
class ToolEvent:
    record_type: str
    sequence_id: int
    seed: int
    template_id: str
    variant_id: str
    config_hash: str
    agent_id: Optional[int]
    tool_name: str
    arguments: Dict[str, Any]
    valid: bool
    invalid_reason: Optional[str]
    invalid_detail: Optional[str]
    abstraction: str
    progress_state: str
    outcome: Optional[str]
    tool_output: Dict[str, Any]
    fixed_defects: List[str]
    active_regressions: List[str]
    applied_patch_ids: List[str]
    step_count: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_patch_text(patch: str) -> str:
    return "\n".join(line.rstrip() for line in patch.strip().splitlines())


def _hash_payload(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _build_patch(file_path: str, old_text: str, new_text: str) -> str:
    return (
        f"*** Update File: {file_path}\n"
        f"- {old_text}\n"
        f"+ {new_text}"
    )


def _build_task_variant(
    *,
    template_id: str,
    variant_id: str,
    title: str,
    description: str,
    source_path: str,
    primary_symbol: str,
    helper_symbol: str,
    primary_old: str,
    primary_new: str,
    helper_old: str,
    helper_new: str,
) -> BugfixTaskVariant:
    primary_defect = f"{template_id}.{variant_id}.primary"
    helper_defect = f"{template_id}.{variant_id}.helper"
    primary_line = 2
    helper_line = 5
    source_content = "\n".join(
        [
            f"def {primary_symbol}(value):",
            f"    {primary_old}",
            "",
            f"def {helper_symbol}(items):",
            f"    {helper_old}",
            "",
        ]
    )
    test_path = f"tests/test_{template_id}_{variant_id}.py"
    test_content = "\n".join(
        [
            f"from src.{template_id}_{variant_id} import {primary_symbol}, {helper_symbol}",
            "",
            f"def test_{template_id}_{variant_id}_targeted():",
            "    assert True",
            "",
            f"def test_{template_id}_{variant_id}_full():",
            "    assert True",
            "",
        ]
    )

    primary_patch = PatchSpec(
        patch_id=f"{template_id}:{variant_id}:primary",
        patch=_build_patch(source_path, primary_old, primary_new),
        file_path=source_path,
        old_text=primary_old,
        new_text=primary_new,
        fixes_defects=(primary_defect,),
        description="Fix the primary defect for the variant.",
    )
    helper_patch = PatchSpec(
        patch_id=f"{template_id}:{variant_id}:helper",
        patch=_build_patch(source_path, helper_old, helper_new),
        file_path=source_path,
        old_text=helper_old,
        new_text=helper_new,
        fixes_defects=(helper_defect,),
        description="Fix the helper defect for the variant.",
    )

    return BugfixTaskVariant(
        template_id=template_id,
        variant_id=variant_id,
        title=title,
        description=description,
        files={source_path: source_content, test_path: test_content},
        symbols={
            primary_symbol: [{"path": source_path, "line": primary_line}],
            helper_symbol: [{"path": source_path, "line": helper_line}],
        },
        patches=(primary_patch, helper_patch),
        test_targets=(
            TestTargetSpec(
                name=f"{test_path}::targeted",
                checks_defects=(primary_defect,),
                description="Checks whether the primary defect is resolved.",
            ),
            TestTargetSpec(
                name=f"{test_path}::full",
                checks_defects=(primary_defect, helper_defect),
                description="Checks the full ticket and regression guards.",
            ),
        ),
        required_defects=(primary_defect, helper_defect),
    )


def build_bugfix_task_fixtures() -> List[BugfixTaskVariant]:
    specs = [
        {
            "template_id": "math_ops",
            "variants": [
                {
                    "variant_id": "sum_bounds",
                    "title": "Fix addition and clamp behavior",
                    "description": "The primary function subtracts instead of adds, and the helper drops the last item.",
                    "source_path": "src/math_ops_sum_bounds.py",
                    "primary_symbol": "compute_total",
                    "helper_symbol": "window_values",
                    "primary_old": "return value - 4",
                    "primary_new": "return value + 4",
                    "helper_old": "return items[:-1]",
                    "helper_new": "return items[:]",
                },
                {
                    "variant_id": "offset_scale",
                    "title": "Fix scale offset ticket",
                    "description": "The primary function scales in the wrong direction, and the helper reverses sorted order.",
                    "source_path": "src/math_ops_offset_scale.py",
                    "primary_symbol": "scale_offset",
                    "helper_symbol": "sorted_window",
                    "primary_old": "return value / 2",
                    "primary_new": "return value * 2",
                    "helper_old": "return sorted(items, reverse=True)",
                    "helper_new": "return sorted(items)",
                },
            ],
        },
        {
            "template_id": "string_utils",
            "variants": [
                {
                    "variant_id": "title_case",
                    "title": "Fix title normalization",
                    "description": "The primary function lowercases everything, and the helper removes interior spaces.",
                    "source_path": "src/string_utils_title_case.py",
                    "primary_symbol": "normalize_title",
                    "helper_symbol": "preserve_spacing",
                    "primary_old": "return value.lower()",
                    "primary_new": "return value.title()",
                    "helper_old": "return ''.join(items)",
                    "helper_new": "return ' '.join(items)",
                },
                {
                    "variant_id": "slugify",
                    "title": "Fix slug normalization",
                    "description": "The primary function leaves spaces untouched, and the helper strips valid dashes.",
                    "source_path": "src/string_utils_slugify.py",
                    "primary_symbol": "slugify_label",
                    "helper_symbol": "normalize_dash",
                    "primary_old": "return value.strip()",
                    "primary_new": "return value.strip().replace(' ', '-')",
                    "helper_old": "return items.replace('-', '')",
                    "helper_new": "return items.replace('--', '-')",
                },
            ],
        },
        {
            "template_id": "retry_policy",
            "variants": [
                {
                    "variant_id": "max_attempts",
                    "title": "Fix retry attempt logic",
                    "description": "The primary function stops too early, and the helper treats transient failures as permanent.",
                    "source_path": "src/retry_policy_max_attempts.py",
                    "primary_symbol": "should_retry",
                    "helper_symbol": "classify_error",
                    "primary_old": "return value < 1",
                    "primary_new": "return value <= 1",
                    "helper_old": "return 'permanent'",
                    "helper_new": "return 'transient'",
                },
                {
                    "variant_id": "backoff_cap",
                    "title": "Fix retry cap handling",
                    "description": "The primary function ignores the cap, and the helper resets attempt counts.",
                    "source_path": "src/retry_policy_backoff_cap.py",
                    "primary_symbol": "next_backoff",
                    "helper_symbol": "record_attempt",
                    "primary_old": "return value * 4",
                    "primary_new": "return min(value * 2, 30)",
                    "helper_old": "return []",
                    "helper_new": "return items + ['attempt']",
                },
            ],
        },
        {
            "template_id": "flag_parser",
            "variants": [
                {
                    "variant_id": "bool_flags",
                    "title": "Fix boolean flag parsing",
                    "description": "The primary function treats 'false' as truthy, and the helper ignores default values.",
                    "source_path": "src/flag_parser_bool_flags.py",
                    "primary_symbol": "parse_enabled",
                    "helper_symbol": "resolve_default",
                    "primary_old": "return bool(value)",
                    "primary_new": "return str(value).lower() in {'1', 'true', 'yes'}",
                    "helper_old": "return None",
                    "helper_new": "return items[0] if items else False",
                },
                {
                    "variant_id": "port_bounds",
                    "title": "Fix port bound parsing",
                    "description": "The primary function parses the wrong base, and the helper allows invalid zero ports.",
                    "source_path": "src/flag_parser_port_bounds.py",
                    "primary_symbol": "parse_port",
                    "helper_symbol": "validate_port",
                    "primary_old": "return int(value, 16)",
                    "primary_new": "return int(value, 10)",
                    "helper_old": "return items >= 0",
                    "helper_new": "return items > 0",
                },
            ],
        },
        {
            "template_id": "billing",
            "variants": [
                {
                    "variant_id": "discount_order",
                    "title": "Fix discount computation order",
                    "description": "The primary function adds tax before discount, and the helper truncates cents.",
                    "source_path": "src/billing_discount_order.py",
                    "primary_symbol": "apply_discount",
                    "helper_symbol": "round_cents",
                    "primary_old": "return value + 5",
                    "primary_new": "return value - 5",
                    "helper_old": "return int(items)",
                    "helper_new": "return round(items, 2)",
                },
                {
                    "variant_id": "invoice_total",
                    "title": "Fix invoice total aggregation",
                    "description": "The primary function drops the final item, and the helper negates credits.",
                    "source_path": "src/billing_invoice_total.py",
                    "primary_symbol": "invoice_total",
                    "helper_symbol": "apply_credit",
                    "primary_old": "return sum(value[:-1])",
                    "primary_new": "return sum(value)",
                    "helper_old": "return items * -1",
                    "helper_new": "return abs(items)",
                },
            ],
        },
        {
            "template_id": "scheduler",
            "variants": [
                {
                    "variant_id": "weekday_window",
                    "title": "Fix weekday scheduler window",
                    "description": "The primary function shifts one day backward, and the helper marks weekdays as weekends.",
                    "source_path": "src/scheduler_weekday_window.py",
                    "primary_symbol": "next_window",
                    "helper_symbol": "is_weekday",
                    "primary_old": "return value - 1",
                    "primary_new": "return value + 1",
                    "helper_old": "return items >= 5",
                    "helper_new": "return items < 5",
                },
                {
                    "variant_id": "cron_merge",
                    "title": "Fix cron merge semantics",
                    "description": "The primary function discards the first schedule, and the helper duplicates merged entries.",
                    "source_path": "src/scheduler_cron_merge.py",
                    "primary_symbol": "merge_schedule",
                    "helper_symbol": "dedupe_entries",
                    "primary_old": "return value[1:]",
                    "primary_new": "return value",
                    "helper_old": "return items + items",
                    "helper_new": "return list(dict.fromkeys(items))",
                },
            ],
        },
        {
            "template_id": "ranking",
            "variants": [
                {
                    "variant_id": "descending_sort",
                    "title": "Fix ranking sort direction",
                    "description": "The primary function returns the lowest score, and the helper sorts ascending.",
                    "source_path": "src/ranking_descending_sort.py",
                    "primary_symbol": "top_score",
                    "helper_symbol": "sort_scores",
                    "primary_old": "return min(value)",
                    "primary_new": "return max(value)",
                    "helper_old": "return sorted(items)",
                    "helper_new": "return sorted(items, reverse=True)",
                },
                {
                    "variant_id": "tie_break",
                    "title": "Fix ranking tie break",
                    "description": "The primary function picks the last tie, and the helper strips duplicate candidates.",
                    "source_path": "src/ranking_tie_break.py",
                    "primary_symbol": "pick_winner",
                    "helper_symbol": "candidate_order",
                    "primary_old": "return value[-1]",
                    "primary_new": "return value[0]",
                    "helper_old": "return sorted(set(items))",
                    "helper_new": "return list(items)",
                },
            ],
        },
        {
            "template_id": "inventory",
            "variants": [
                {
                    "variant_id": "stock_count",
                    "title": "Fix stock count threshold",
                    "description": "The primary function reports out-of-stock items as available, and the helper ignores the final bin.",
                    "source_path": "src/inventory_stock_count.py",
                    "primary_symbol": "is_available",
                    "helper_symbol": "count_bins",
                    "primary_old": "return value <= 0",
                    "primary_new": "return value > 0",
                    "helper_old": "return sum(items[:-1])",
                    "helper_new": "return sum(items)",
                },
                {
                    "variant_id": "reserve_logic",
                    "title": "Fix reserve allocation logic",
                    "description": "The primary function subtracts reserve twice, and the helper clamps to zero too early.",
                    "source_path": "src/inventory_reserve_logic.py",
                    "primary_symbol": "allocatable_units",
                    "helper_symbol": "clamp_inventory",
                    "primary_old": "return value - 2",
                    "primary_new": "return value - 1",
                    "helper_old": "return 0",
                    "helper_new": "return max(items, 0)",
                },
            ],
        },
    ]

    variants: List[BugfixTaskVariant] = []
    for template in specs:
        for variant in template["variants"]:
            variants.append(_build_task_variant(template_id=template["template_id"], **variant))
    return variants


class ToolUseBugfixEnvironment:
    def __init__(
        self,
        variant: BugfixTaskVariant,
        *,
        seed: int,
        num_agents: int = 2,
        step_budget: int = 12,
    ):
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if step_budget <= 0:
            raise ValueError("step_budget must be positive")

        self.variant = variant
        self.seed = seed
        self.num_agents = num_agents
        self.step_budget = step_budget
        self.progress_state = NOT_STARTED
        self.outcome: Optional[str] = None
        self.step_count = 0
        self.event_sequence = 0
        self.current_files = copy.deepcopy(variant.files)
        self.applied_patch_ids: List[str] = []
        self.fixed_defects: List[str] = []
        self.active_regressions: List[str] = []
        self.observed_files: List[str] = []
        self.observed_symbols: List[str] = []
        self.executed_tests: List[str] = []
        self.invalid_call_counts = {
            INVALID_SCHEMA: 0,
            INVALID_REFERENCE: 0,
            INVALID_STATE: 0,
            INVALID_NOOP: 0,
            INVALID_BUDGET: 0,
        }

        config_payload = {
            "environment_version": ENVIRONMENT_VERSION,
            "template_id": variant.template_id,
            "variant_id": variant.variant_id,
            "seed": seed,
            "num_agents": num_agents,
            "step_budget": step_budget,
        }
        self.config_hash = _hash_payload(config_payload)

        self.patch_lookup = {
            _normalize_patch_text(patch.patch): patch for patch in self.variant.patches
        }
        self.test_lookup = {target.name: target for target in self.variant.test_targets}

    @classmethod
    def from_seed(
        cls,
        *,
        seed: int,
        num_agents: int = 2,
        step_budget: int = 12,
        fixtures: Optional[Sequence[BugfixTaskVariant]] = None,
        template_id: Optional[str] = None,
        variant_id: Optional[str] = None,
    ) -> "ToolUseBugfixEnvironment":
        fixture_set = list(fixtures or build_bugfix_task_fixtures())
        if not fixture_set:
            raise ValueError("At least one fixture is required")

        if template_id and variant_id:
            for candidate in fixture_set:
                if candidate.template_id == template_id and candidate.variant_id == variant_id:
                    return cls(candidate, seed=seed, num_agents=num_agents, step_budget=step_budget)
            raise ValueError(f"Unknown fixture {template_id}/{variant_id}")

        rng = random.Random(seed)
        selected = fixture_set[rng.randrange(len(fixture_set))]
        return cls(selected, seed=seed, num_agents=num_agents, step_budget=step_budget)

    def export_episode_header(self) -> Dict[str, Any]:
        return {
            "record_type": "episode_header",
            "schema_version": SCHEMA_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "template_id": self.variant.template_id,
            "variant_id": self.variant.variant_id,
            "title": self.variant.title,
            "description": self.variant.description,
            "num_agents": self.num_agents,
            "step_budget": self.step_budget,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def export_replay_spec(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "template_id": self.variant.template_id,
            "variant_id": self.variant.variant_id,
            "step_budget": self.step_budget,
            "num_agents": self.num_agents,
            "environment_version": ENVIRONMENT_VERSION,
            "config_hash": self.config_hash,
        }

    def list_valid_actions(self) -> List[Dict[str, Any]]:
        if self.is_terminal() or self.step_count >= self.step_budget:
            return []

        actions: List[Dict[str, Any]] = []
        for path in sorted(self.current_files.keys()):
            actions.append({"tool_name": "retrieve_file", "arguments": {"path": path}})

        for symbol_name in sorted(self.variant.symbols.keys()):
            actions.append({"tool_name": "search_symbol", "arguments": {"name": symbol_name}})

        actions.append({"tool_name": "run_tests", "arguments": {}})
        for target_name in sorted(self.test_lookup.keys()):
            actions.append({"tool_name": "run_tests", "arguments": {"test_target": target_name}})

        for patch_spec in self.variant.patches:
            if patch_spec.patch_id in self.applied_patch_ids:
                continue
            current_content = self.current_files.get(patch_spec.file_path, "")
            if patch_spec.old_text not in current_content:
                continue
            actions.append({"tool_name": "apply_patch", "arguments": {"patch": patch_spec.patch}})

        if self.applied_patch_ids or self.executed_tests:
            actions.append(
                {
                    "tool_name": "finalize_ticket",
                    "arguments": {"reason": "Episode review complete."},
                }
            )

        return actions

    def is_terminal(self) -> bool:
        return self.progress_state in {COMPLETED_SUCCESS, COMPLETED_PARTIAL, COMPLETED_FAILURE}

    def step(self, agent_id: int, tool_name: str, arguments: Dict[str, Any]) -> ToolEvent:
        if tool_name not in VALID_TOOL_NAMES:
            return self._emit_event(
                agent_id=agent_id,
                tool_name=tool_name,
                arguments=arguments,
                valid=False,
                invalid_reason=INVALID_SCHEMA,
                invalid_detail="unknown_tool_name",
                abstraction="error_or_noop",
                tool_output={"message": "Unknown tool"},
            )

        if self.is_terminal():
            return self._emit_event(
                agent_id=agent_id,
                tool_name=tool_name,
                arguments=arguments,
                valid=False,
                invalid_reason=INVALID_STATE,
                invalid_detail="episode_already_completed",
                abstraction="error_or_noop",
                tool_output={"message": "Episode already completed"},
            )

        if self.step_count >= self.step_budget:
            self._auto_finalize_for_budget()
            return self._emit_event(
                agent_id=agent_id,
                tool_name=tool_name,
                arguments=arguments,
                valid=False,
                invalid_reason=INVALID_BUDGET,
                invalid_detail="step_budget_exhausted",
                abstraction="error_or_noop",
                tool_output={"message": "Step budget exhausted"},
            )

        handler = getattr(self, f"_handle_{tool_name}")
        valid, invalid_reason, invalid_detail, abstraction, tool_output = handler(arguments)

        self.step_count += 1
        event = self._emit_event(
            agent_id=agent_id,
            tool_name=tool_name,
            arguments=arguments,
            valid=valid,
            invalid_reason=invalid_reason,
            invalid_detail=invalid_detail,
            abstraction=abstraction,
            tool_output=tool_output,
            increment_step=False,
        )

        if not self.is_terminal() and self.step_count >= self.step_budget:
            self._auto_finalize_for_budget()
            event.progress_state = self.progress_state
            event.outcome = self.outcome
            event.tool_output = {
                **event.tool_output,
                "budget_exhausted": True,
                "final_outcome": self.outcome,
            }

        return event

    def summarize_episode(self, events: Sequence[ToolEvent]) -> Dict[str, Any]:
        full_suite = self._evaluate_tests([target.name for target in self.variant.test_targets])
        passes = sum(1 for result in full_suite if result["passed"])
        return {
            "schema_version": SCHEMA_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "template_id": self.variant.template_id,
            "variant_id": self.variant.variant_id,
            "progress_state": self.progress_state,
            "outcome": self.outcome,
            "step_count": self.step_count,
            "step_budget": self.step_budget,
            "invalid_call_counts": self.invalid_call_counts,
            "invalid_call_rate": sum(self.invalid_call_counts.values()) / self.step_count if self.step_count else 0.0,
            "applied_patch_ids": list(self.applied_patch_ids),
            "fixed_defects": list(self.fixed_defects),
            "full_test_results": full_suite,
            "tests_passed": passes,
            "tests_total": len(full_suite),
            "event_count": len(events),
            "replay_spec": self.export_replay_spec(),
        }

    def validate_abstraction_invariants(self, events: Sequence[ToolEvent]) -> None:
        allowed = {"retrieve", "verify", "update", "finalize", "error_or_noop"}
        for event in events:
            if event.abstraction not in allowed:
                raise AssertionError(f"Unexpected abstraction category: {event.abstraction}")

    def _handle_retrieve_file(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], str, Dict[str, Any]]:
        schema_error = self._validate_arguments(arguments, required={"path": str}, optional={"span": dict})
        if schema_error:
            return False, INVALID_SCHEMA, schema_error, "error_or_noop", {"message": "Invalid retrieve_file arguments"}

        path = arguments["path"]
        if path not in self.current_files:
            return False, INVALID_REFERENCE, "unknown_file_path", "error_or_noop", {"message": "File not found"}

        content = self.current_files[path].splitlines()
        start_line = 1
        end_line = len(content)
        span = arguments.get("span")
        if span is not None:
            if set(span.keys()) != {"start", "end"}:
                return False, INVALID_SCHEMA, "span_requires_start_and_end", "error_or_noop", {"message": "Invalid span"}
            if not isinstance(span["start"], int) or not isinstance(span["end"], int):
                return False, INVALID_SCHEMA, "span_values_must_be_int", "error_or_noop", {"message": "Invalid span types"}
            if span["start"] < 1 or span["end"] < span["start"]:
                return False, INVALID_SCHEMA, "span_bounds_invalid", "error_or_noop", {"message": "Invalid span bounds"}
            start_line = span["start"]
            end_line = min(span["end"], len(content))

        if path not in self.observed_files:
            self.observed_files.append(path)
        self._mark_in_progress()
        return True, None, None, "retrieve", {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "content": "\n".join(content[start_line - 1:end_line]),
        }

    def _handle_search_symbol(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], str, Dict[str, Any]]:
        schema_error = self._validate_arguments(arguments, required={"name": str}, optional={})
        if schema_error:
            return False, INVALID_SCHEMA, schema_error, "error_or_noop", {"message": "Invalid search_symbol arguments"}

        name = arguments["name"]
        matches = self.variant.symbols.get(name)
        if not matches:
            return False, INVALID_REFERENCE, "unknown_symbol_name", "error_or_noop", {"message": "Symbol not found"}

        if name not in self.observed_symbols:
            self.observed_symbols.append(name)
        self._mark_in_progress()
        return True, None, None, "retrieve", {"matches": matches, "count": len(matches)}

    def _handle_run_tests(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], str, Dict[str, Any]]:
        schema_error = self._validate_arguments(arguments, required={}, optional={"test_target": str})
        if schema_error:
            return False, INVALID_SCHEMA, schema_error, "error_or_noop", {"message": "Invalid run_tests arguments"}

        requested = arguments.get("test_target")
        if requested is None:
            targets = [target.name for target in self.variant.test_targets]
        else:
            if requested not in self.test_lookup:
                return False, INVALID_REFERENCE, "unknown_test_target", "error_or_noop", {"message": "Test target not found"}
            targets = [requested]

        self._mark_in_progress()
        for target in targets:
            if target not in self.executed_tests:
                self.executed_tests.append(target)

        results = self._evaluate_tests(targets)
        return True, None, None, "verify", {
            "results": results,
            "passed": sum(1 for result in results if result["passed"]),
            "failed": sum(1 for result in results if not result["passed"]),
        }

    def _handle_apply_patch(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], str, Dict[str, Any]]:
        schema_error = self._validate_arguments(arguments, required={"patch": str}, optional={})
        if schema_error:
            return False, INVALID_SCHEMA, schema_error, "error_or_noop", {"message": "Invalid apply_patch arguments"}

        patch_text = arguments["patch"].strip()
        if not patch_text:
            return False, INVALID_NOOP, "empty_patch", "error_or_noop", {"message": "Patch was empty"}

        normalized = _normalize_patch_text(patch_text)
        patch_spec = self.patch_lookup.get(normalized)
        if patch_spec is None:
            return False, INVALID_REFERENCE, "unknown_patch", "error_or_noop", {"message": "Patch not recognized"}

        if patch_spec.patch_id in self.applied_patch_ids:
            return False, INVALID_NOOP, "patch_already_applied", "error_or_noop", {"message": "Patch already applied"}

        current_content = self.current_files[patch_spec.file_path]
        if patch_spec.old_text not in current_content:
            if patch_spec.new_text in current_content:
                return False, INVALID_NOOP, "same_content_patch", "error_or_noop", {"message": "Patch would not change content"}
            return False, INVALID_REFERENCE, "patch_context_missing", "error_or_noop", {"message": "Patch context missing"}

        self.current_files[patch_spec.file_path] = current_content.replace(patch_spec.old_text, patch_spec.new_text, 1)
        self.applied_patch_ids.append(patch_spec.patch_id)
        for defect in patch_spec.fixes_defects:
            if defect not in self.fixed_defects:
                self.fixed_defects.append(defect)
        for regression in patch_spec.introduces_regressions:
            if regression not in self.active_regressions:
                self.active_regressions.append(regression)

        self._mark_in_progress()
        return True, None, None, "update", {
            "patch_id": patch_spec.patch_id,
            "file_path": patch_spec.file_path,
            "fixed_defects": list(patch_spec.fixes_defects),
        }

    def _handle_finalize_ticket(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], str, Dict[str, Any]]:
        schema_error = self._validate_arguments(arguments, required={"reason": str}, optional={})
        if schema_error:
            return False, INVALID_SCHEMA, schema_error, "error_or_noop", {"message": "Invalid finalize_ticket arguments"}

        reason = arguments["reason"].strip()
        if not reason:
            return False, INVALID_SCHEMA, "reason_must_be_non_empty", "error_or_noop", {"message": "Reason cannot be empty"}

        if not self.applied_patch_ids and not self.executed_tests:
            return False, INVALID_STATE, "finalize_requires_update_or_verify", "error_or_noop", {
                "message": "Finalize requires at least one update or verify step",
            }

        self.outcome = self._determine_outcome(reason_present=True, budget_exhausted=False)
        self.progress_state = self._progress_state_for_outcome(self.outcome)
        return True, None, None, "finalize", {
            "reason": reason,
            "final_outcome": self.outcome,
            "required_defects_fixed": len(self.fixed_defects),
        }

    def _evaluate_tests(self, targets: Sequence[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        fixed_defects = set(self.fixed_defects)
        regressions = set(self.active_regressions)
        for target_name in targets:
            target = self.test_lookup[target_name]
            missing = [defect for defect in target.checks_defects if defect not in fixed_defects]
            failing_regressions = [guard for guard in target.regression_guards if guard in regressions]
            results.append(
                {
                    "test_target": target_name,
                    "passed": not missing and not failing_regressions,
                    "missing_defects": missing,
                    "failing_regressions": failing_regressions,
                }
            )
        return results

    def _determine_outcome(self, *, reason_present: bool, budget_exhausted: bool) -> str:
        fixed_required = set(self.fixed_defects).intersection(self.variant.required_defects)
        all_required_fixed = len(fixed_required) == len(self.variant.required_defects)
        full_results = self._evaluate_tests([target.name for target in self.variant.test_targets])
        all_tests_pass = all(result["passed"] for result in full_results)
        regression_failures = sum(len(result["failing_regressions"]) for result in full_results)
        catastrophic_regression = regression_failures > self.variant.max_regression_failures

        if all_required_fixed and all_tests_pass and not catastrophic_regression:
            return OUTCOME_SUCCESS

        if fixed_required and not all_tests_pass and not catastrophic_regression and (reason_present or budget_exhausted):
            return OUTCOME_PARTIAL

        return OUTCOME_FAILURE

    def _auto_finalize_for_budget(self) -> None:
        if self.is_terminal():
            return
        self.outcome = self._determine_outcome(reason_present=False, budget_exhausted=True)
        self.progress_state = self._progress_state_for_outcome(self.outcome)

    def _mark_in_progress(self) -> None:
        if self.progress_state == NOT_STARTED:
            self.progress_state = IN_PROGRESS

    def _emit_event(
        self,
        *,
        agent_id: Optional[int],
        tool_name: str,
        arguments: Dict[str, Any],
        valid: bool,
        invalid_reason: Optional[str],
        invalid_detail: Optional[str],
        abstraction: str,
        tool_output: Dict[str, Any],
        increment_step: bool = True,
    ) -> ToolEvent:
        if invalid_reason:
            self.invalid_call_counts[invalid_reason] += 1
        if increment_step:
            self.step_count += 1

        self.event_sequence += 1
        event = ToolEvent(
            record_type="tool_event",
            sequence_id=self.event_sequence,
            seed=self.seed,
            template_id=self.variant.template_id,
            variant_id=self.variant.variant_id,
            config_hash=self.config_hash,
            agent_id=agent_id,
            tool_name=tool_name,
            arguments=copy.deepcopy(arguments),
            valid=valid,
            invalid_reason=invalid_reason,
            invalid_detail=invalid_detail,
            abstraction=abstraction,
            progress_state=self.progress_state,
            outcome=self.outcome,
            tool_output=copy.deepcopy(tool_output),
            fixed_defects=list(self.fixed_defects),
            active_regressions=list(self.active_regressions),
            applied_patch_ids=list(self.applied_patch_ids),
            step_count=self.step_count,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        return event

    def _progress_state_for_outcome(self, outcome: str) -> str:
        if outcome == OUTCOME_SUCCESS:
            return COMPLETED_SUCCESS
        if outcome == OUTCOME_PARTIAL:
            return COMPLETED_PARTIAL
        return COMPLETED_FAILURE

    def _validate_arguments(
        self,
        arguments: Dict[str, Any],
        *,
        required: Dict[str, type],
        optional: Dict[str, type],
    ) -> Optional[str]:
        if not isinstance(arguments, dict):
            return "arguments_must_be_object"
        allowed_keys = set(required.keys()).union(optional.keys())
        if set(arguments.keys()) - allowed_keys:
            return "unexpected_argument_keys"
        for key, expected_type in required.items():
            if key not in arguments:
                return f"missing_required_argument:{key}"
            if not isinstance(arguments[key], expected_type):
                return f"wrong_type_for_argument:{key}"
        for key, expected_type in optional.items():
            if key in arguments and not isinstance(arguments[key], expected_type):
                return f"wrong_type_for_argument:{key}"
        return None


def build_demo_actions(environment: ToolUseBugfixEnvironment) -> List[Tuple[int, str, Dict[str, Any]]]:
    source_path = next(path for path in environment.variant.files if path.startswith("src/"))
    primary_symbol = next(iter(environment.variant.symbols.keys()))
    primary_patch, helper_patch = environment.variant.patches
    full_target = next(target.name for target in environment.variant.test_targets if target.name.endswith("::full"))
    targeted_target = next(target.name for target in environment.variant.test_targets if target.name.endswith("::targeted"))
    second_agent = 1 if environment.num_agents > 1 else 0
    return [
        (0, "retrieve_file", {"path": source_path, "span": {"start": 1, "end": 6}}),
        (second_agent, "search_symbol", {"name": primary_symbol}),
        (0, "apply_patch", {"patch": primary_patch.patch}),
        (second_agent, "run_tests", {"test_target": targeted_target}),
        (second_agent, "apply_patch", {"patch": helper_patch.patch}),
        (0, "run_tests", {"test_target": full_target}),
        (0, "finalize_ticket", {"reason": "Targeted tests passed after both required fixes were applied."}),
    ]


def run_demo_episode(*, seed: int, num_agents: int = 2, step_budget: int = 12) -> Tuple[ToolUseBugfixEnvironment, List[ToolEvent]]:
    environment = ToolUseBugfixEnvironment.from_seed(seed=seed, num_agents=num_agents, step_budget=step_budget)
    events: List[ToolEvent] = []
    for agent_id, tool_name, arguments in build_demo_actions(environment):
        events.append(environment.step(agent_id, tool_name, arguments))
        if environment.is_terminal():
            break
    environment.validate_abstraction_invariants(events)
    return environment, events


def persist_episode_run(
    *,
    output_root: str,
    environment: ToolUseBugfixEnvironment,
    events: Sequence[ToolEvent],
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        output_root,
        f"tool_use_run_{environment.variant.template_id}_{environment.variant.variant_id}_{timestamp}",
    )
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "episode.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(environment.export_episode_header()) + "\n")
        for event in events:
            handle.write(json.dumps(event.to_dict()) + "\n")

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(environment.summarize_episode(events), handle, indent=2)

    with open(os.path.join(run_dir, "task_fixture.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(environment.variant), handle, indent=2)

    return run_dir