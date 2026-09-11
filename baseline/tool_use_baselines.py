from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llm_client import (
    DEFAULT_ACTION_PARSE_ATTEMPTS,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    chat_json,
)

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
BASELINE_LLM_SINGLE_AGENT = "llm_single_agent"
BASELINE_LLM_INDEPENDENT_MULTI_AGENT = "llm_independent_multi_agent"
BASELINE_LLM_PROMPTED_COORDINATION = "llm_prompted_coordination"

VALID_BASELINES = {
    BASELINE_SINGLE_AGENT,
    BASELINE_RANDOM_POLICY,
    BASELINE_INDEPENDENT_MULTI_AGENT,
    BASELINE_PROMPTED_COORDINATION,
    BASELINE_LLM_SINGLE_AGENT,
    BASELINE_LLM_INDEPENDENT_MULTI_AGENT,
    BASELINE_LLM_PROMPTED_COORDINATION,
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


class _LLMPolicyBase(BasePolicyAdapter):
    def __init__(
        self,
        *,
        seed: int,
        num_agents: int,
        model_name: Optional[str] = None,
        llm_call: Optional[Callable[[str], Any]] = None,
    ):
        super().__init__(seed=seed, num_agents=num_agents)
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.llm_call = llm_call or self._call_model
        self.parse_attempts = max(DEFAULT_ACTION_PARSE_ATTEMPTS, 1)
        self.action_history: List[PolicyAction] = []

    def _call_model(self, prompt: str) -> Any:
        return chat_json(
            model=self.model_name,
            prompt=prompt,
            max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        )

    def _build_prompt(
        self,
        environment: ToolUseBugfixEnvironment,
        *,
        role_instructions: str,
        include_agent_id: bool,
    ) -> str:
        state = {
            "task": {
                "title": environment.variant.title,
                "description": environment.variant.description,
                "template_id": environment.variant.template_id,
                "variant_id": environment.variant.variant_id,
            },
            "progress": {
                "progress_state": environment.progress_state,
                "step_count": environment.step_count,
                "step_budget": environment.step_budget,
                "observed_files": list(environment.observed_files),
                "observed_symbols": list(environment.observed_symbols),
                "executed_tests": list(environment.executed_tests),
                "applied_patch_ids": list(environment.applied_patch_ids),
                "fixed_defects": list(environment.fixed_defects),
                "active_regressions": list(environment.active_regressions),
            },
            "recent_actions": [
                {
                    "agent_id": action.agent_id,
                    "tool_name": action.tool_name,
                    "arguments": action.arguments,
                }
                for action in self.action_history[-6:]
            ],
            "agent_ids": list(range(environment.num_agents)),
            "valid_actions": environment.list_valid_actions(),
        }
        response_shape = {
            "agent_id": "integer in agent_ids",
            "tool_name": "one tool_name from valid_actions",
            "arguments": "the exact arguments object from the matching valid action",
        }
        if not include_agent_id:
            response_shape.pop("agent_id")
        return (
            f"{role_instructions}\n"
            "Choose exactly one action from valid_actions. Match both tool_name and "
            "arguments exactly; do not invent tools, arguments, paths, symbols, tests, or patches.\n"
            "Make task progress: do not repeat an exact recent action when another valid action "
            "is available. Do not retrieve an already observed file or search an already observed "
            "symbol. Prefer a valid apply_patch that addresses a described defect; after required "
            "patches, run relevant tests and then finalize the ticket.\n"
            "Return only a JSON object with this shape:\n"
            f"{json.dumps(response_shape, sort_keys=True)}\n\n"
            "Current task and environment state:\n"
            f"{json.dumps(state, sort_keys=True, indent=2)}"
        )

    def _select_action(
        self,
        environment: ToolUseBugfixEnvironment,
        *,
        preferred_agent_id: int,
        role_instructions: str,
        include_agent_id: bool,
    ) -> Optional[PolicyAction]:
        available = environment.list_valid_actions()
        if not available:
            return None

        base_prompt = self._build_prompt(
            environment,
            role_instructions=role_instructions,
            include_agent_id=include_agent_id,
        )
        retry_note = ""
        for _ in range(self.parse_attempts):
            try:
                response = self.llm_call(f"{base_prompt}{retry_note}")
                if getattr(response, "is_fallback", False):
                    break
                payload = self._response_payload(response)
                action = self._match_action(
                    payload,
                    available,
                    preferred_agent_id=preferred_agent_id,
                    include_agent_id=include_agent_id,
                    environment=environment,
                )
                if action is not None:
                    if self._is_unproductive_repeat(action, available, environment):
                        raise ValueError("response repeated an action that should not be repeated")
                    self.action_history.append(action)
                    return action
                raise ValueError("response did not match a currently valid action")
            except (TypeError, ValueError) as exc:
                retry_note = (
                    "\n\nThe previous response was invalid: "
                    f"{exc}. Return only one exact currently valid action as JSON."
                )
            except Exception as exc:
                retry_note = (
                    "\n\nThe previous model call failed: "
                    f"{type(exc).__name__}: {exc}. Return one exact valid action."
                )

        fallback = self._fallback_action_without_recent_repeat(
            environment,
            preferred_agent_id=preferred_agent_id,
        )
        if fallback is not None:
            self.action_history.append(fallback)
        return fallback

    def _is_unproductive_repeat(
        self,
        action: PolicyAction,
        available: Sequence[Dict[str, Any]],
        environment: ToolUseBugfixEnvironment,
    ) -> bool:
        action_key = (action.tool_name, _freeze_arguments(action.arguments))
        previous_keys = {
            (item.tool_name, _freeze_arguments(item.arguments)) for item in self.action_history
        }
        if action_key not in previous_keys:
            has_patch_action = any(item["tool_name"] == "apply_patch" for item in available)
            if action.tool_name == "retrieve_file":
                return action.arguments.get("path") in environment.observed_files or (
                    bool(environment.observed_files) and has_patch_action
                )
            if action.tool_name == "search_symbol":
                return action.arguments.get("name") in environment.observed_symbols or (
                    bool(environment.observed_symbols) and has_patch_action
                )
            return False

        available_keys = {
            (item["tool_name"], _freeze_arguments(item["arguments"])) for item in available
        }
        return any(candidate not in previous_keys for candidate in available_keys)

    def _fallback_action_without_recent_repeat(
        self,
        environment: ToolUseBugfixEnvironment,
        *,
        preferred_agent_id: int,
    ) -> Optional[PolicyAction]:
        available = environment.list_valid_actions()
        previous_keys = {
            (item.tool_name, _freeze_arguments(item.arguments)) for item in self.action_history
        }
        if environment.observed_files or environment.observed_symbols:
            for item in available:
                candidate_key = (item["tool_name"], _freeze_arguments(item["arguments"]))
                if item["tool_name"] == "apply_patch" and candidate_key not in previous_keys:
                    return PolicyAction(
                        agent_id=min(preferred_agent_id, max(environment.num_agents - 1, 0)),
                        tool_name=item["tool_name"],
                        arguments=dict(item["arguments"]),
                    )
        for item in available:
            candidate_key = (item["tool_name"], _freeze_arguments(item["arguments"]))
            if candidate_key in previous_keys:
                continue
            return PolicyAction(
                agent_id=min(preferred_agent_id, max(environment.num_agents - 1, 0)),
                tool_name=item["tool_name"],
                arguments=dict(item["arguments"]),
            )
        return self._fallback_action(environment, preferred_agent_id=preferred_agent_id)

    @staticmethod
    def _response_payload(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            payload = response
        else:
            content = response
            choices = getattr(response, "choices", None)
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None)
            if isinstance(content, dict):
                payload = content
            elif isinstance(content, str):
                payload = json.loads(content)
            else:
                raise ValueError("response did not contain JSON content")

        if not isinstance(payload, dict):
            raise ValueError("JSON response must be an object")
        return payload

    @staticmethod
    def _match_action(
        payload: Dict[str, Any],
        available: Sequence[Dict[str, Any]],
        *,
        preferred_agent_id: int,
        include_agent_id: bool,
        environment: ToolUseBugfixEnvironment,
    ) -> Optional[PolicyAction]:
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            raise ValueError("tool_name must be a string and arguments must be an object")

        agent_id = preferred_agent_id
        if include_agent_id and "agent_id" in payload:
            requested_agent = payload["agent_id"]
            if isinstance(requested_agent, bool) or not isinstance(requested_agent, int):
                raise ValueError("agent_id must be an integer")
            if requested_agent < 0 or requested_agent >= environment.num_agents:
                raise ValueError("agent_id is not in the current agent_ids list")
            agent_id = requested_agent

        requested_key = (tool_name, _freeze_arguments(arguments))
        for action in available:
            action_key = (action["tool_name"], _freeze_arguments(action["arguments"]))
            if requested_key == action_key:
                return PolicyAction(agent_id, tool_name, dict(arguments))
        return None


class LLMSingleAgentPolicy(_LLMPolicyBase):
    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        return self._select_action(
            environment,
            preferred_agent_id=0,
            role_instructions=(
                "You are the single software agent responsible for completing the bugfix ticket "
                "within the remaining step budget."
            ),
            include_agent_id=False,
        )


class LLMIndependentMultiAgentPolicy(_LLMPolicyBase):
    def __init__(
        self,
        *,
        seed: int,
        num_agents: int,
        model_name: Optional[str] = None,
        llm_call: Optional[Callable[[str], Any]] = None,
    ):
        super().__init__(seed=seed, num_agents=num_agents, model_name=model_name, llm_call=llm_call)
        self.turn_index = 0

    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        agent_id = self.turn_index % max(environment.num_agents, 1)
        self.turn_index += 1
        return self._select_action(
            environment,
            preferred_agent_id=agent_id,
            role_instructions=(
                f"You are independent agent {agent_id}. Choose the next useful action from the "
                "shared environment state without assuming another agent will perform it."
            ),
            include_agent_id=False,
        )


class LLMPromptedCoordinationPolicy(_LLMPolicyBase):
    def __init__(
        self,
        *,
        seed: int,
        num_agents: int,
        model_name: Optional[str] = None,
        llm_call: Optional[Callable[[str], Any]] = None,
    ):
        super().__init__(seed=seed, num_agents=num_agents, model_name=model_name, llm_call=llm_call)
        self.turn_index = 0

    def next_action(self, environment: ToolUseBugfixEnvironment) -> Optional[PolicyAction]:
        preferred_agent_id = self.turn_index % max(environment.num_agents, 1)
        self.turn_index += 1
        return self._select_action(
            environment,
            preferred_agent_id=preferred_agent_id,
            role_instructions=(
                "You are a shared coordinator. Choose the next useful action and assign it to "
                "the agent best positioned to perform it."
            ),
            include_agent_id=True,
        )


def build_policy_adapter(
    baseline_name: str,
    *,
    seed: int,
    num_agents: int,
    model_name: Optional[str] = None,
    llm_call: Optional[Callable[[str], Any]] = None,
) -> BasePolicyAdapter:
    if baseline_name == BASELINE_SINGLE_AGENT:
        return SingleAgentPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_RANDOM_POLICY:
        return RandomValidPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_INDEPENDENT_MULTI_AGENT:
        return IndependentMultiAgentPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_PROMPTED_COORDINATION:
        return PromptedCoordinationPolicy(seed=seed, num_agents=num_agents)
    if baseline_name == BASELINE_LLM_SINGLE_AGENT:
        return LLMSingleAgentPolicy(seed=seed, num_agents=num_agents, model_name=model_name, llm_call=llm_call)
    if baseline_name == BASELINE_LLM_INDEPENDENT_MULTI_AGENT:
        return LLMIndependentMultiAgentPolicy(
            seed=seed,
            num_agents=num_agents,
            model_name=model_name,
            llm_call=llm_call,
        )
    if baseline_name == BASELINE_LLM_PROMPTED_COORDINATION:
        return LLMPromptedCoordinationPolicy(
            seed=seed,
            num_agents=num_agents,
            model_name=model_name,
            llm_call=llm_call,
        )
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
    model_name: Optional[str] = None,
    llm_call: Optional[Callable[[str], Any]] = None,
) -> Tuple[ToolUseBugfixEnvironment, List[ToolEvent]]:
    if baseline_name not in VALID_BASELINES:
        raise ValueError(f"Unknown baseline name: {baseline_name}")

    episode_agents = 1 if baseline_name in {BASELINE_SINGLE_AGENT, BASELINE_LLM_SINGLE_AGENT} else max(num_agents, 2)
    environment = ToolUseBugfixEnvironment.from_seed(
        seed=seed,
        num_agents=episode_agents,
        step_budget=step_budget,
        fixtures=fixtures,
        template_id=template_id,
        variant_id=variant_id,
    )
    policy = build_policy_adapter(
        baseline_name,
        seed=seed,
        num_agents=episode_agents,
        model_name=model_name,
        llm_call=llm_call,
    )
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