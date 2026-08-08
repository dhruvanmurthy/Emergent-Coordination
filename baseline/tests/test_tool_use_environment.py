import unittest

from tool_use_environment import (
    COMPLETED_PARTIAL,
    COMPLETED_SUCCESS,
    IN_PROGRESS,
    INVALID_NOOP,
    INVALID_STATE,
    NOT_STARTED,
    OUTCOME_PARTIAL,
    OUTCOME_SUCCESS,
    ToolUseBugfixEnvironment,
    build_bugfix_task_fixtures,
)


class ToolUseEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.variant = build_bugfix_task_fixtures()[0]
        self.environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=11,
            num_agents=2,
            step_budget=6,
        )

    def test_retrieve_transitions_to_in_progress(self):
        source_path = next(path for path in self.variant.files if path.startswith("src/"))
        self.assertEqual(self.environment.progress_state, NOT_STARTED)

        event = self.environment.step(0, "retrieve_file", {"path": source_path})

        self.assertTrue(event.valid)
        self.assertEqual(self.environment.progress_state, IN_PROGRESS)
        self.assertEqual(event.abstraction, "retrieve")

    def test_finalize_before_update_or_verify_is_invalid(self):
        event = self.environment.step(0, "finalize_ticket", {"reason": "No-op"})

        self.assertFalse(event.valid)
        self.assertEqual(event.invalid_reason, INVALID_STATE)
        self.assertEqual(self.environment.progress_state, NOT_STARTED)

    def test_success_requires_both_required_patches(self):
        primary_patch, helper_patch = self.variant.patches
        full_target = next(target.name for target in self.variant.test_targets if target.name.endswith("::full"))

        self.environment.step(0, "apply_patch", {"patch": primary_patch.patch})
        self.environment.step(1, "apply_patch", {"patch": helper_patch.patch})
        self.environment.step(0, "run_tests", {"test_target": full_target})
        event = self.environment.step(0, "finalize_ticket", {"reason": "All required fixes landed."})

        self.assertTrue(event.valid)
        self.assertEqual(self.environment.outcome, OUTCOME_SUCCESS)
        self.assertEqual(self.environment.progress_state, COMPLETED_SUCCESS)

    def test_budget_exhaustion_marks_partial_when_progress_exists(self):
        primary_patch = self.variant.patches[0]
        targeted_target = next(target.name for target in self.variant.test_targets if target.name.endswith("::targeted"))

        partial_environment = ToolUseBugfixEnvironment(
            self.variant,
            seed=11,
            num_agents=2,
            step_budget=2,
        )
        partial_environment.step(0, "apply_patch", {"patch": primary_patch.patch})
        last_event = partial_environment.step(1, "run_tests", {"test_target": targeted_target})

        self.assertEqual(partial_environment.outcome, OUTCOME_PARTIAL)
        self.assertEqual(partial_environment.progress_state, COMPLETED_PARTIAL)
        self.assertTrue(last_event.tool_output["budget_exhausted"])

    def test_reapplying_the_same_patch_is_a_semantic_noop(self):
        primary_patch = self.variant.patches[0]

        self.environment.step(0, "apply_patch", {"patch": primary_patch.patch})
        event = self.environment.step(1, "apply_patch", {"patch": primary_patch.patch})

        self.assertFalse(event.valid)
        self.assertEqual(event.invalid_reason, INVALID_NOOP)


if __name__ == "__main__":
    unittest.main()