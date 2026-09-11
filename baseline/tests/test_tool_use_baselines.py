import unittest

from tool_use_baselines import (
    BASELINE_LLM_INDEPENDENT_MULTI_AGENT,
    BASELINE_LLM_PROMPTED_COORDINATION,
    BASELINE_LLM_SINGLE_AGENT,
    BASELINE_INDEPENDENT_MULTI_AGENT,
    BASELINE_PROMPTED_COORDINATION,
    BASELINE_RANDOM_POLICY,
    BASELINE_SINGLE_AGENT,
    LLMIndependentMultiAgentPolicy,
    LLMPromptedCoordinationPolicy,
    LLMSingleAgentPolicy,
    build_policy_adapter,
    run_policy_episode,
)
from tool_use_environment import (
    OUTCOME_FAILURE,
    OUTCOME_PARTIAL,
    OUTCOME_SUCCESS,
    ToolUseBugfixEnvironment,
    build_bugfix_task_fixtures,
)


class ToolUseBaselineTests(unittest.TestCase):
    def setUp(self):
        self.variant = build_bugfix_task_fixtures()[0]

    def test_valid_actions_gate_finalize_until_progress_exists(self):
        environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=13,
            num_agents=2,
            step_budget=8,
        )

        initial_actions = environment.list_valid_actions()
        self.assertNotIn("finalize_ticket", {action["tool_name"] for action in initial_actions})

        primary_patch = self.variant.patches[0]
        environment.step(0, "apply_patch", {"patch": primary_patch.patch})

        later_actions = environment.list_valid_actions()
        self.assertIn("finalize_ticket", {action["tool_name"] for action in later_actions})

    def test_single_agent_policy_reaches_success(self):
        environment, events = run_policy_episode(
            baseline_name=BASELINE_SINGLE_AGENT,
            seed=17,
            step_budget=12,
            fixtures=[self.variant],
            template_id=self.variant.template_id,
            variant_id=self.variant.variant_id,
        )

        self.assertEqual(environment.outcome, OUTCOME_SUCCESS)
        self.assertTrue(events)

    def test_independent_multi_agent_policy_reaches_success(self):
        environment, _ = run_policy_episode(
            baseline_name=BASELINE_INDEPENDENT_MULTI_AGENT,
            seed=19,
            step_budget=12,
            fixtures=[self.variant],
            template_id=self.variant.template_id,
            variant_id=self.variant.variant_id,
        )

        self.assertEqual(environment.outcome, OUTCOME_SUCCESS)

    def test_prompted_coordination_policy_reaches_success(self):
        environment, _ = run_policy_episode(
            baseline_name=BASELINE_PROMPTED_COORDINATION,
            seed=23,
            step_budget=12,
            fixtures=[self.variant],
            template_id=self.variant.template_id,
            variant_id=self.variant.variant_id,
        )

        self.assertEqual(environment.outcome, OUTCOME_SUCCESS)

    def test_random_policy_stays_within_outcome_space(self):
        environment, events = run_policy_episode(
            baseline_name=BASELINE_RANDOM_POLICY,
            seed=29,
            step_budget=12,
            fixtures=[self.variant],
            template_id=self.variant.template_id,
            variant_id=self.variant.variant_id,
        )

        self.assertTrue(events)
        self.assertIn(environment.outcome, {OUTCOME_SUCCESS, OUTCOME_PARTIAL, OUTCOME_FAILURE})

    def test_single_agent_is_partial_on_hard_variant(self):
        hard_variant = next(
            variant
            for variant in build_bugfix_task_fixtures()
            if variant.template_id == "billing" and variant.variant_id == "invoice_total"
        )
        environment, _ = run_policy_episode(
            baseline_name=BASELINE_SINGLE_AGENT,
            seed=31,
            step_budget=8,
            fixtures=[hard_variant],
            template_id=hard_variant.template_id,
            variant_id=hard_variant.variant_id,
        )

        self.assertEqual(environment.outcome, OUTCOME_PARTIAL)

    def test_llm_single_agent_retries_and_matches_a_valid_json_action(self):
        source_path = next(path for path in self.variant.files if path.startswith("src/"))
        responses = [
            '{"tool_name": "invented_tool", "arguments": {}}',
            '{"tool_name": "retrieve_file", "arguments": {"path": "' + source_path + '"}}',
        ]
        prompts = []

        def fake_llm(prompt):
            prompts.append(prompt)
            return responses.pop(0)

        environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=41,
            num_agents=1,
            step_budget=8,
        )
        policy = LLMSingleAgentPolicy(seed=41, num_agents=1, llm_call=fake_llm)

        action = policy.next_action(environment)

        self.assertEqual(action.tool_name, "retrieve_file")
        self.assertEqual(action.arguments, {"path": source_path})
        self.assertEqual(len(prompts), 2)
        self.assertIn(self.variant.title, prompts[0])
        self.assertIn("valid_actions", prompts[0])

    def test_llm_policy_falls_back_after_hallucinated_actions(self):
        environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=43,
            num_agents=1,
            step_budget=8,
        )

        def fake_llm(_prompt):
            return '{"tool_name": "invented_tool", "arguments": {}}'

        policy = LLMSingleAgentPolicy(seed=43, num_agents=1, llm_call=fake_llm)
        action = policy.next_action(environment)
        valid_keys = {
            (item["tool_name"], tuple(sorted(item["arguments"].items())))
            for item in environment.list_valid_actions()
        }

        self.assertIn((action.tool_name, tuple(sorted(action.arguments.items()))), valid_keys)

    def test_llm_policy_rejects_retrieval_loop_after_observation(self):
        source_path = next(path for path in self.variant.files if path.startswith("src/"))
        responses = [
            {
                "tool_name": "retrieve_file",
                "arguments": {"path": source_path},
            },
            {
                "tool_name": "retrieve_file",
                "arguments": {"path": source_path},
            },
        ]

        def fake_llm(_prompt):
            return responses.pop(0)

        environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=45,
            num_agents=1,
            step_budget=8,
        )
        policy = LLMSingleAgentPolicy(seed=45, num_agents=1, llm_call=fake_llm)

        first_action = policy.next_action(environment)
        environment.step(first_action.agent_id, first_action.tool_name, first_action.arguments)
        second_action = policy.next_action(environment)

        self.assertEqual(first_action.tool_name, "retrieve_file")
        self.assertEqual(second_action.tool_name, "apply_patch")

    def test_llm_coordinator_accepts_the_assigned_agent(self):
        source_path = next(path for path in self.variant.files if path.startswith("src/"))
        environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=47,
            num_agents=2,
            step_budget=8,
        )

        def fake_llm(_prompt):
            return {
                "agent_id": 1,
                "tool_name": "retrieve_file",
                "arguments": {"path": source_path},
            }

        policy = LLMPromptedCoordinationPolicy(seed=47, num_agents=2, llm_call=fake_llm)
        action = policy.next_action(environment)

        self.assertEqual(action.agent_id, 1)
        self.assertEqual(action.tool_name, "retrieve_file")

    def test_llm_baseline_names_build_expected_policy_types(self):
        policy_types = {
            BASELINE_LLM_SINGLE_AGENT: LLMSingleAgentPolicy,
            BASELINE_LLM_INDEPENDENT_MULTI_AGENT: LLMIndependentMultiAgentPolicy,
            BASELINE_LLM_PROMPTED_COORDINATION: LLMPromptedCoordinationPolicy,
        }
        for baseline_name, policy_type in policy_types.items():
            policy = build_policy_adapter(
                baseline_name,
                seed=53,
                num_agents=2,
                llm_call=lambda _prompt: "{}",
            )
            self.assertIsInstance(policy, policy_type)


if __name__ == "__main__":
    unittest.main()