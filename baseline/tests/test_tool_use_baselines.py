import unittest

from tool_use_baselines import (
    BASELINE_INDEPENDENT_MULTI_AGENT,
    BASELINE_PROMPTED_COORDINATION,
    BASELINE_RANDOM_POLICY,
    BASELINE_SINGLE_AGENT,
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


if __name__ == "__main__":
    unittest.main()