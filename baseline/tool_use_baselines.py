from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tool_use_environment import (
    INVALID_NOOP,
    OUTCOME_FAILURE,
    OUTCOME_PARTIAL,
    OUTCOME_SUCCESS,
    ToolEvent,
    ToolUseBugfixEnvironment,
    build_demo_actions,
)


BASELINE_SINGLE_AGENT = "single_agent"
BASELINE_RANDOM_POLICY = "random_policy"
BASELINE_INDEPENDENT_MULTI_AGENT = "independent_multi_agent"
BASELINE_PROMPTED_COORDINATION = "prompted_coordination"

VALID_BASELINES = {
    BASELINE_SINGLE_AGENT,
    BASELINE_RANDOM_POLICY,
    BASELINE_INDEPENDENT_MULTI_AGENT,
    BASELINE_PROMPTED_COORDINATION,
}


@dataclass(frozen=True)
class PolicyAction:
    agent_id: int
    tool_name: str
    arguments: Dict[str, Any]


def _freeze_arguments(arguments: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    def freeze_value(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((key, freeze_value(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(freeze_value(item) for item in value))
        return value

    return tuple(sorted((key, freeze_value(value)) for key, value in arguments.items()))


class BasePolicyAdapter:
    def __init__(self, *, seed: int, num_agents: int):
        self.seed = seed
        self.num_agents = num_agents
        self.random = random.Random(seed)

    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        raise NotImplementedError

    def _fallback_action(
        self,
        environment: ToolUseBugfixEnvironment,
        *,
        preferred_agent_id: int = 0,
    ) -> Optional[PolicyAction]:
        available = environment.list_valid_actions()
        if not available:
            return None

        finalize_action = next(
            (action for action in available if action["tool_name"] == "finalize_ticket"),
            None,
        )
        selected = finalize_action or available[0]
        return PolicyAction(
            agent_id=min(preferred_agent_id, max(environment.num_agents - 1, 0)),
            tool_name=selected["tool_name"],
            arguments=dict(selected["arguments"]),
        )


class SingleAgentPolicy(BasePolicyAdapter):
    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        source_path = next(path for path in environment.variant.files if path.startswith("src/"))
        primary_symbol = next(iter(environment.variant.symbols.keys()))
        primary_patch = environment.variant.patches[0]
        helper_patch = environment.variant.patches[1]
        targeted_target = next(
            target.name for target in environment.variant.test_targets if target.name.endswith("::targeted")
        )
        full_target = next(
            target.name for target in environment.variant.test_targets if target.name.endswith("::full")
        )

        if source_path not in environment.observed_files:
            return PolicyAction(0, "retrieve_file", {"path": source_path})
        if primary_symbol not in environment.observed_symbols:
            return PolicyAction(0, "search_symbol", {"name": primary_symbol})
        if primary_patch.patch_id not in environment.applied_patch_ids:
            return PolicyAction(0, "apply_patch", {"patch": primary_patch.patch})
        if targeted_target not in environment.executed_tests:
            return PolicyAction(0, "run_tests", {"test_target": targeted_target})
        if helper_patch.patch_id not in environment.applied_patch_ids:
            return PolicyAction(0, "apply_patch", {"patch": helper_patch.patch})
        if full_target not in environment.executed_tests:
            return PolicyAction(0, "run_tests", {"test_target": full_target})

        return self._fallback_action(environment, preferred_agent_id=0)


class RandomValidPolicy(BasePolicyAdapter):
    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        available = environment.list_valid_actions()
        if not available:
            return None

        selected = self.random.choice(available)
        return PolicyAction(
            agent_id=self.random.randrange(environment.num_agents),
            tool_name=selected["tool_name"],
            arguments=dict(selected["arguments"]),
        )


class IndependentMultiAgentPolicy(BasePolicyAdapter):
    def __init__(self, *, seed: int, num_agents: int):
        super().__init__(seed=seed, num_agents=num_agents)
        self.turn_index = 0
        self.agent_plans: Optional[Dict[int, List[PolicyAction]]] = None
        self.agent_offsets: Dict[int, int] = {}

    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        if self.agent_plans is None:
            self.agent_plans = self._build_agent_plans(environment)
            self.agent_offsets = {agent_id: 0 for agent_id in self.agent_plans}

        available = environment.list_valid_actions()
        if not available:
            return None

        available_keys = {
            (action["tool_name"], _freeze_arguments(action["arguments"])) for action in available
        }

        for _ in range(max(environment.num_agents, 1)):
            agent_id = self.turn_index % environment.num_agents
            self.turn_index += 1
            plan = self.agent_plans.get(agent_id, [])
            offset = self.agent_offsets.get(agent_id, 0)
            while offset < len(plan):
                candidate = plan[offset]
                self.agent_offsets[agent_id] = offset + 1
                offset += 1
                candidate_key = (candidate.tool_name, _freeze_arguments(candidate.arguments))
                if candidate_key in available_keys:
                    return candidate

        return self._fallback_action(environment, preferred_agent_id=0)

    def _build_agent_plans(self, environment: ToolUseBugfixEnvironment) -> Dict[int, List[PolicyAction]]:
        source_path = next(path for path in environment.variant.files if path.startswith("src/"))
        primary_symbol = next(iter(environment.variant.symbols.keys()))
        primary_patch = environment.variant.patches[0]
        helper_patch = environment.variant.patches[1]
        targeted_target = next(
            target.name for target in environment.variant.test_targets if target.name.endswith("::targeted")
        )
        full_target = next(
            target.name for target in environment.variant.test_targets if target.name.endswith("::full")
        )

        plans: Dict[int, List[PolicyAction]] = {
            0: [
                PolicyAction(0, "retrieve_file", {"path": source_path}),
                PolicyAction(0, "apply_patch", {"patch": primary_patch.patch}),
                PolicyAction(0, "run_tests", {"test_target": targeted_target}),
                PolicyAction(0, "finalize_ticket", {"reason": "Independent review completed."}),
            ],
        }

        if environment.num_agents > 1:
            plans[1] = [
                PolicyAction(1, "search_symbol", {"name": primary_symbol}),
                PolicyAction(1, "apply_patch", {"patch": helper_patch.patch}),
                PolicyAction(1, "run_tests", {"test_target": full_target}),
            ]

        for agent_id in range(2, environment.num_agents):
            plans[agent_id] = [
                PolicyAction(agent_id, "run_tests", {"test_target": full_target}),
            ]

        return plans

class PromptedCoordinationPolicy(BasePolicyAdapter):
    def __init__(self, *, seed: int, num_agents: int):
        super().__init__(seed=seed, num_agents=num_agents)
        self.plan_index = 0
        self.shared_plan: Optional[List[PolicyAction]] = None

    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        if self.shared_plan is None:
            self.shared_plan = [
                PolicyAction(agent_id, tool_name, dict(arguments))
                for agent_id, tool_name, arguments in build_demo_actions(environment)
            ]

        available = environment.list_valid_actions()
        if not available:
            return None

        available_keys = {
            (action["tool_name"], _freeze_arguments(action["arguments"])) for action in available
        }
        while self.plan_index < len(self.shared_plan):
            candidate = self.shared_plan[self.plan_index]
            self.plan_index += 1
            candidate_key = (candidate.tool_name, _freeze_arguments(candidate.arguments))
            if candidate_key in available_keys:
                return candidate

        return self._fallback_action(environment, preferred_agent_id=0)

def build_policy_adapter(baseline_name: str, *, seed: int, num_agents: int) -> BasePolicyAdapter:
    if baseline_name == BASELINE_SINGLE_AGENT:
        return SingleAgentPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_RANDOM_POLICY:
        return RandomValidPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_INDEPENDENT_MULTI_AGENT:
        return IndependentMultiAgentPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_PROMPTED_COORDINATION:
        return PromptedCoordinationPolicy(seed=seed, num_agents=num_agents)
    raise ValueError(f"Unknown baseline policy: {baseline_name}")


def run_policy_episode(
    *,
    baseline_name: str,
    seed: int,
    step_budget: int,
    num_agents: int = 2,
    fixtures: Optional[Sequence[Any]] = None,
    template_id: Optional[str] = None,
    variant_id: Optional[str] = None,
) -> Tuple[ToolUseBugfixEnvironment, List[ToolEvent]]:
    if baseline_name not in VALID_BASELINES:
        raise ValueError(f"Unknown baseline name: {baseline_name}")

    episode_agents = 1 if baseline_name == BASELINE_SINGLE_AGENT else max(num_agents, 2)
    environment = ToolUseBugfixEnvironment.from_seed(
        seed=seed,
        num_agents=episode_agents,
        step_budget=step_budget,
        fixtures=fixtures,
        template_id=template_id,
        variant_id=variant_id,
    )
    policy = build_policy_adapter(baseline_name, seed=seed, num_agents=episode_agents)
    events: List[ToolEvent] = []

    while not environment.is_terminal():
        action = policy.next_action(environment)
        if action is None:
            break
        events.append(environment.step(action.agent_id, action.tool_name, action.arguments))

    environment.validate_abstraction_invariants(events)
    return environment, events


def summarize_baseline_episode(
    *,
    baseline_name: str,
    environment: ToolUseBugfixEnvironment,
    events: Sequence[ToolEvent],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    summary = environment.summarize_episode(events)
    tool_counts = Counter(event.tool_name for event in events)
    abstraction_counts = Counter(event.abstraction for event in events)
    accepted_patches = sum(1 for event in events if event.tool_name == "apply_patch" and event.valid)
    patch_attempts = tool_counts.get("apply_patch", 0)
    test_attempts = tool_counts.get("run_tests", 0)

    summary.update(
        {
            "baseline_name": baseline_name,
            "model_name": model_name,
            "success": environment.outcome == OUTCOME_SUCCESS,
            "partial": environment.outcome == OUTCOME_PARTIAL,
            "failure": environment.outcome == OUTCOME_FAILURE,
            "tool_counts": dict(tool_counts),
            "abstraction_counts": dict(abstraction_counts),
            "calls_to_completion": len(events),
            "test_run_efficiency": test_attempts / len(events) if events else 0.0,
            "patch_acceptance_ratio": accepted_patches / patch_attempts if patch_attempts else 0.0,
            "semantic_noop_count": environment.invalid_call_counts[INVALID_NOOP],
        }
    )
    return summary