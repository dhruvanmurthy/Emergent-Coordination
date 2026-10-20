from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import replay_tool_actions
from tool_action_environment import (
    ACTION_CATEGORY,
    ACTION_NAME,
    COMPLETED,
    OUTCOME_SOLVED,
    OUTCOME_UNSOLVED,
    Phase1GuessingEnvironment,
)


class Phase1ToolActionEnvironmentTests(unittest.TestCase):
    def make_environment(
        self,
        *,
        mode="sum",
        target=10,
        agent_ids=(0, 1),
        max_rounds=3,
        guess_range=(0, 50),
        target_range=(0, 100),
    ):
        return Phase1GuessingEnvironment(
            mode=mode,
            target=target,
            target_range=target_range,
            guess_range=guess_range,
            agent_ids=agent_ids,
            max_rounds=max_rounds,
        )

    @staticmethod
    def submit_round(environment, guesses):
        events = []
        for agent_id, value in guesses.items():
            environment.observe(agent_id)
            events.append(environment.step(agent_id, ACTION_NAME, {"value": value}))
        return events

    def test_sum_mode_aggregates_all_guesses_and_marks_contribution(self):
        environment = self.make_environment(target=7)
        self.assertEqual(
            environment.list_valid_actions(),
            [
                {
                    "tool_name": ACTION_NAME,
                    "arguments_schema": {
                        "value": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 50,
                        }
                    },
                }
            ],
        )

        events = self.submit_round(environment, {0: 3, 1: 4})

        result = environment._round_results[0]
        self.assertEqual(result["guesses"], {0: 3, 1: 4})
        self.assertEqual(result["average"], 7)
        self.assertEqual(result["rounded_average"], 7)
        self.assertEqual(result["comparison_value"], 7)
        self.assertEqual(result["feedback"], "CORRECT! ")
        self.assertTrue(all(event.valid for event in events))
        self.assertEqual(events[0].agent_id, 0)
        self.assertEqual(events[0].action_category, ACTION_CATEGORY)
        self.assertEqual(events[0].tool_name, "submit_guess")
        self.assertEqual(events[0].round_num, 1)
        self.assertEqual(events[0].tool_output, {"accepted": True})
        self.assertEqual(environment._outcome, OUTCOME_SOLVED)

    def test_submit_guess_enforces_the_configured_range(self):
        environment = self.make_environment(
            target=10,
            guess_range=(0, 10),
        )

        rejected = environment.step(0, ACTION_NAME, {"value": 11})

        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.invalid_reason, "schema_invalid")
        self.assertEqual(rejected.invalid_detail, "value_out_of_range")
        self.assertEqual(rejected.action_category, ACTION_CATEGORY)
        self.assertEqual(environment._round_results, [])
        self.assertEqual(environment._step_count, 1)

    def test_mean_mode_uses_python_rounding_and_stops_when_solved(self):
        environment = self.make_environment(mode="mean", target=2, max_rounds=4)

        first_round = self.submit_round(environment, {0: 0, 1: 1})
        self.assertEqual(first_round[-1].round_num, 1)
        self.assertEqual(environment._round_results[0]["average"], 0.5)
        self.assertEqual(environment._round_results[0]["rounded_average"], 0)
        self.assertEqual(environment._round_results[0]["feedback"], "too LOW ")

        second_round = self.submit_round(environment, {0: 1, 1: 2})
        self.assertTrue(all(event.round_num == 2 for event in second_round))
        self.assertEqual(environment._round_results[1]["average"], 1.5)
        self.assertEqual(environment._round_results[1]["rounded_average"], 2)
        self.assertEqual(environment._round_results[1]["feedback"], "CORRECT! ")
        self.assertEqual(environment._outcome, OUTCOME_SOLVED)
        self.assertEqual(environment._execution_status, "completed")
        self.assertEqual(environment._progress_state, COMPLETED)
        self.assertEqual(len(environment._round_results), 2)

        rejected = environment.step(0, ACTION_NAME, {"value": 2})
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.invalid_reason, "state_invalid")

    def test_round_limit_is_distinct_from_tool_call_budget(self):
        environment = self.make_environment(target=1, max_rounds=7)

        for _ in range(7):
            events = self.submit_round(environment, {0: 0, 1: 0})

        self.assertTrue(all(event.valid for event in events))
        self.assertEqual(environment._outcome, OUTCOME_UNSOLVED)
        self.assertEqual(environment._execution_status, "completed")
        self.assertEqual(environment._progress_state, COMPLETED)
        self.assertEqual(len(environment._round_results), 7)
        self.assertEqual(environment._step_count, 14)
        self.assertEqual(environment._round_results[-1]["feedback"], "too LOW ")

    def test_no_feedback_or_private_data_is_exposed_before_round_completion(self):
        environment = self.make_environment(
            target=10,
            agent_ids=(0, 1, 2),
            guess_range=(0, 50),
        )
        self.assertFalse(hasattr(environment, "target"))
        self.assertFalse(hasattr(environment, "round_results"))
        self.assertFalse(hasattr(environment, "agent_ids"))
        with self.assertRaises(ValueError):
            environment.observe(True)
        initial_observation = environment.observe(0)
        self.assertEqual(
            set(initial_observation),
            {"round_num", "mode", "guess_range", "goal", "history"},
        )
        self.assertEqual(initial_observation["history"], [])

        first_event = environment.step(0, ACTION_NAME, {"value": 5})
        self.assertEqual(first_event.tool_output, {"accepted": True})
        for agent_id in (0, 1, 2):
            observation = environment.observe(agent_id)
            self.assertEqual(observation["history"], [])
            self.assertFalse(
                {
                    "agent_ids",
                    "group_size",
                    "mystery_number",
                    "num_agents",
                    "peer_guesses",
                    "pending_guesses",
                    "target",
                }.intersection(observation)
            )
        self.assertEqual(environment._round_results, [])

        environment.step(1, ACTION_NAME, {"value": 3})
        self.assertEqual(environment.observe(1)["history"], [])
        self.assertEqual(environment._round_results, [])
        environment.step(2, ACTION_NAME, {"value": 2})

        observations = [environment.observe(agent_id) for agent_id in (0, 1, 2)]
        self.assertTrue(
            all(observation["history"][0]["feedback"] == "CORRECT! " for observation in observations)
        )
        self.assertEqual(observations[0]["history"][0]["own_guess"], 5)
        self.assertEqual(observations[1]["history"][0]["own_guess"], 3)
        self.assertEqual(observations[2]["history"][0]["own_guess"], 2)
        self.assertEqual(
            set(observations[0]["history"][0]),
            {"round_num", "own_guess", "feedback"},
        )

    def test_next_round_requires_agent_to_observe_shared_feedback(self):
        environment = self.make_environment(target=1, max_rounds=3)
        self.submit_round(environment, {0: 0, 1: 0})

        rejected = environment.step(0, ACTION_NAME, {"value": 1})
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.invalid_detail, "feedback_not_observed")

        environment.observe(0)
        accepted = environment.step(0, ACTION_NAME, {"value": 1})
        self.assertTrue(accepted.valid)
        self.assertEqual(accepted.round_num, 2)

    def test_replay_cli_returns_nonzero_when_report_has_mismatches(self):
        failed_report = {
            "passed": False,
            "case_count": 2,
            "trajectory_count": 1,
            "rounds_checked": 22,
            "mismatches": [{"case_id": "reference", "field": "feedback"}],
        }
        with patch.object(
            replay_tool_actions,
            "run_replay",
            return_value=(Path("report.json"), failed_report),
        ):
            with redirect_stdout(StringIO()):
                exit_code = replay_tool_actions.main(["--output-dir", "unused"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
