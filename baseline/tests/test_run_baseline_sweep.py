import json
import os
import shutil
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

from llm_client import (
    FallbackResponse,
    LLMRequestError,
)
from run_baseline_sweep import (
    ArtifactValidationError,
    EpisodeCollectionError,
    _persist_episode_error,
    build_episode_manifest,
    load_runner_settings,
    run_batch,
    run_for_model,
    run_single_episode,
)
from tool_use_baselines import (
    BASELINE_LLM_SINGLE_AGENT,
    BASELINE_SINGLE_AGENT,
    LLMFallbackError,
    LLMResponseError,
    run_policy_episode,
)
from tool_use_environment import build_bugfix_task_fixtures


class RunBaselineSweepTests(unittest.TestCase):
    def _single_agent_config(self, seeds):
        return {
            "environment": "synthetic_bugfix",
            "models": ["offline-test-model"],
            "baselines": [BASELINE_SINGLE_AGENT],
            "num_agents": 2,
            "step_budget": 12,
            "episode_seeds": list(seeds),
            "fixture_repetitions": None,
            "seed_start": 200,
            "batch_size": 1,
            "results_prefix": "test",
            "strict_llm": False,
        }

    @staticmethod
    def _error_for_call(args, kwargs, error):
        item = {
            "manifest_id": kwargs["manifest_id"],
            "baseline": args[0],
            "repetition": kwargs["repetition"],
            "seed": args[1],
            "template_id": kwargs["template_id"],
            "variant_id": kwargs["variant_id"],
        }
        error_path = _persist_episode_error(args[2], item, error)
        return EpisodeCollectionError(item, error_path, error)

    @staticmethod
    def _fake_completed_result(item):
        return {
            "manifest_id": item["manifest_id"],
            "baseline_name": item["baseline"],
            "collection_status": "completed",
            "outcome": "success",
            "success": True,
            "partial": False,
            "failure": False,
            "invalid_call_rate": 0.0,
            "calls_to_completion": 1,
            "patch_acceptance_ratio": 1.0,
        }

    @staticmethod
    def _read_json(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def _run_batch_with_error(self, error, *, continue_on_retryable_failure):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_episode_manifest(
                [BASELINE_SINGLE_AGENT],
                episode_seeds=[200, 201],
            )
            calls = []

            def fake_run_single(*args, **kwargs):
                calls.append(kwargs["manifest_id"])
                if len(calls) == 1:
                    raise self._error_for_call(args, kwargs, error)
                return self._fake_completed_result(manifest[1])

            with patch("run_baseline_sweep.run_single_episode", side_effect=fake_run_single):
                if continue_on_retryable_failure:
                    progress = run_batch(
                        manifest,
                        1,
                        1,
                        temp_dir,
                        "offline-test-model",
                        2,
                        12,
                        0.0,
                        completed_ids=set(),
                        total_items=len(manifest),
                        continue_on_retryable_failure=True,
                    )
                else:
                    with self.assertRaises(EpisodeCollectionError):
                        run_batch(
                            manifest,
                            1,
                            1,
                            temp_dir,
                            "offline-test-model",
                            2,
                            12,
                            0.0,
                            completed_ids=set(),
                            total_items=len(manifest),
                            continue_on_retryable_failure=continue_on_retryable_failure,
                        )
                    progress = self._read_json(os.path.join(temp_dir, "progress.json"))
            return progress, calls

    def test_thesis_manifest_is_balanced_and_deterministically_ordered(self):
        cfg = load_runner_settings("thesis_experiment")
        fixtures = build_bugfix_task_fixtures()

        first = build_episode_manifest(
            cfg["baselines"],
            fixtures=fixtures,
            repetitions_per_fixture=cfg["fixture_repetitions"],
            seed_start=cfg["seed_start"],
        )
        second = build_episode_manifest(
            cfg["baselines"],
            fixtures=fixtures,
            repetitions_per_fixture=cfg["fixture_repetitions"],
            seed_start=cfg["seed_start"],
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 192)
        self.assertEqual(Counter(item["baseline"] for item in first), {
            "llm_single_agent": 64,
            "llm_independent_multi_agent": 64,
            "llm_prompted_coordination": 64,
        })
        self.assertEqual(
            set(Counter((item["template_id"], item["variant_id"]) for item in first).values()),
            {12},
        )
        self.assertEqual(
            {(item["template_id"], item["variant_id"]) for item in first},
            {(fixture.template_id, fixture.variant_id) for fixture in fixtures},
        )
        self.assertTrue(all({
            "baseline",
            "repetition",
            "seed",
            "template_id",
            "variant_id",
        }.issubset(item) for item in first))

    def test_explicit_fixture_replay_overrides_seed_selection(self):
        fixtures = build_bugfix_task_fixtures()
        first_fixture = fixtures[0]
        last_fixture = fixtures[-1]

        first_environment, _ = run_policy_episode(
            baseline_name=BASELINE_SINGLE_AGENT,
            seed=200,
            step_budget=12,
            template_id=first_fixture.template_id,
            variant_id=first_fixture.variant_id,
        )
        last_environment, _ = run_policy_episode(
            baseline_name=BASELINE_SINGLE_AGENT,
            seed=200,
            step_budget=12,
            template_id=last_fixture.template_id,
            variant_id=last_fixture.variant_id,
        )

        self.assertEqual(
            (first_environment.variant.template_id, first_environment.variant.variant_id),
            (first_fixture.template_id, first_fixture.variant_id),
        )
        self.assertEqual(
            (last_environment.variant.template_id, last_environment.variant.variant_id),
            (last_fixture.template_id, last_fixture.variant_id),
        )

    def test_resume_skips_completed_manifest_item_without_creating_duplicate(self):
        cfg = {
            "environment": "synthetic_bugfix",
            "models": ["offline-test-model"],
            "baselines": [BASELINE_SINGLE_AGENT],
            "num_agents": 2,
            "step_budget": 12,
            "episode_seeds": [200],
            "fixture_repetitions": None,
            "seed_start": 200,
            "batch_size": 1,
            "results_prefix": "test",
            "strict_llm": False,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "thesis_resume")
            first_summary = run_for_model(
                "offline-test-model",
                cfg,
                run_dir=run_dir,
                max_items=1,
            )
            first_episode_dirs = sorted(os.listdir(os.path.join(run_dir, "episodes")))

            second_summary = run_for_model(
                "offline-test-model",
                cfg,
                run_dir=run_dir,
                resume=True,
                max_items=1,
            )
            second_episode_dirs = sorted(os.listdir(os.path.join(run_dir, "episodes")))

            self.assertEqual(
                first_summary["baseline_aggregates"],
                second_summary["baseline_aggregates"],
            )
            self.assertEqual(first_episode_dirs, second_episode_dirs)
            self.assertEqual(len(second_episode_dirs), 1)
            with open(os.path.join(run_dir, "progress.json"), encoding="utf-8") as handle:
                progress = json.load(handle)
            completed = [
                entry
                for entry in progress["items"].values()
                if entry["status"] == "completed"
            ]
            self.assertEqual(len(completed), 1)

    def test_strict_llm_fallback_raises_instead_of_emitting_action(self):
        fixture = build_bugfix_task_fixtures()[0]

        def fallback_call(_prompt):
            return FallbackResponse(
                model="azureai/gpt-5-mini",
                content='{"tool_name": null, "arguments": {}}',
                reason="test fallback",
            )

        with self.assertRaises(LLMFallbackError):
            run_policy_episode(
                baseline_name=BASELINE_LLM_SINGLE_AGENT,
                seed=211,
                step_budget=12,
                template_id=fixture.template_id,
                variant_id=fixture.variant_id,
                strict_llm=True,
                llm_call=fallback_call,
            )

    def test_strict_llm_failure_persists_manifest_identity_and_reason(self):
        fixture = build_bugfix_task_fixtures()[0]
        manifest_id = "llm_single_agent__rep_001__seed_211__math_ops__sum_bounds"

        def fallback_call(_prompt):
            return FallbackResponse(
                model="azureai/gpt-5-mini",
                content='{"tool_name": null, "arguments": {}}',
                reason="persisted test fallback",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(EpisodeCollectionError) as context:
                run_single_episode(
                    BASELINE_LLM_SINGLE_AGENT,
                    211,
                    temp_dir,
                    "azureai/gpt-5-mini",
                    1,
                    12,
                    template_id=fixture.template_id,
                    variant_id=fixture.variant_id,
                    repetition=1,
                    manifest_id=manifest_id,
                    strict_llm=True,
                    llm_call=fallback_call,
                )

            self.assertTrue(os.path.isfile(context.exception.error_path))
            with open(context.exception.error_path, encoding="utf-8") as handle:
                error = json.load(handle)
            self.assertEqual(error["manifest_id"], manifest_id)
            self.assertEqual(error["fallback_reason"], "persisted test fallback")

    def test_continue_after_isolated_retryable_episode_failure(self):
        progress, calls = self._run_batch_with_error(
            LLMResponseError("final response was invalid"),
            continue_on_retryable_failure=True,
        )

        self.assertEqual(len(calls), 2)
        statuses = {
            manifest_id: entry["status"]
            for manifest_id, entry in progress["items"].items()
        }
        self.assertEqual(list(statuses.values()), ["failed", "completed"])
        self.assertEqual(progress["failed_count"], 1)
        self.assertEqual(progress["failed_manifest_ids"], [calls[0]])
        self.assertEqual(progress["batches_completed"][0]["status"], "failed")
        self.assertEqual(progress["batches_completed"][0]["completed_count"], 1)
        self.assertEqual(progress["batches_completed"][0]["failed_count"], 1)

    def test_retryable_failure_remains_fail_fast_by_default(self):
        progress, calls = self._run_batch_with_error(
            LLMResponseError("final response was invalid"),
            continue_on_retryable_failure=False,
        )

        self.assertEqual(len(calls), 1)
        failure = next(iter(progress["items"].values()))
        self.assertEqual(failure["status"], "failed")
        self.assertTrue(failure["result"]["retryable"])

    def test_structured_policy_audit_can_make_response_failure_fatal(self):
        response_error = LLMResponseError("response was invalid")
        response_error.policy_audit = {"fatal_category": "quota"}
        progress, calls = self._run_batch_with_error(
            response_error,
            continue_on_retryable_failure=False,
        )

        self.assertEqual(len(calls), 1)
        failure = next(iter(progress["items"].values()))
        self.assertEqual(failure["result"]["classification"], "quota_failure")
        self.assertFalse(failure["result"]["retryable"])

    def test_authentication_failure_stops_even_when_continuation_is_enabled(self):
        auth_error = RuntimeError("authentication failed")
        auth_error.status_code = 401
        progress, calls = self._run_batch_with_error(
            auth_error,
            continue_on_retryable_failure=False,
        )

        self.assertEqual(len(calls), 1)
        failure = next(iter(progress["items"].values()))
        self.assertEqual(failure["result"]["classification"], "authentication_failure")
        self.assertEqual(progress["batches_completed"][0]["status"], "aborted")

    def test_quota_failure_stops_even_when_continuation_is_enabled(self):
        quota_error = LLMRequestError(
            provider="azureai",
            model="gpt-5-mini",
            attempt_count=5,
            last_error=RuntimeError("quota response"),
        )
        quota_error.error_category = "quota"
        progress, calls = self._run_batch_with_error(
            quota_error,
            continue_on_retryable_failure=False,
        )

        self.assertEqual(len(calls), 1)
        failure = next(iter(progress["items"].values()))
        self.assertEqual(failure["result"]["classification"], "quota_failure")

    def test_repeated_api_failure_stops_after_client_retries(self):
        repeated_error = LLMRequestError(
            provider="azureai",
            model="gpt-5-mini",
            attempt_count=5,
            last_error=RuntimeError("server response"),
        )
        repeated_error.error_category = "repeated_api_failure"
        progress, calls = self._run_batch_with_error(
            repeated_error,
            continue_on_retryable_failure=False,
        )

        self.assertEqual(len(calls), 1)
        failure = next(iter(progress["items"].values()))
        self.assertEqual(failure["result"]["classification"], "repeated_api_failure")

    def test_strict_fallback_stops_collection(self):
        progress, calls = self._run_batch_with_error(
            LLMFallbackError("strict fallback"),
            continue_on_retryable_failure=False,
        )

        self.assertEqual(len(calls), 1)
        failure = next(iter(progress["items"].values()))
        self.assertEqual(failure["result"]["classification"], "strict_llm_fallback")

    def test_disk_and_dependency_failures_stop_collection(self):
        for error in (OSError("disk full"), ModuleNotFoundError("missing package")):
            with self.subTest(error=type(error).__name__):
                progress, calls = self._run_batch_with_error(
                    error,
                    continue_on_retryable_failure=False,
                )
                self.assertEqual(len(calls), 1)
                failure = next(iter(progress["items"].values()))
                self.assertFalse(failure["result"]["retryable"])

    def test_completed_aggregate_excludes_failed_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = build_episode_manifest(
                [BASELINE_SINGLE_AGENT],
                episode_seeds=[200, 201],
            )
            with patch("run_baseline_sweep.run_single_episode") as mocked:
                mocked.side_effect = [
                    self._fake_completed_result(manifest[0]),
                    self._error_for_call(
                        (
                            BASELINE_SINGLE_AGENT,
                            manifest[1]["seed"],
                            temp_dir,
                            "offline-test-model",
                            2,
                            12,
                        ),
                        {
                            "manifest_id": manifest[1]["manifest_id"],
                            "repetition": manifest[1]["repetition"],
                            "template_id": manifest[1]["template_id"],
                            "variant_id": manifest[1]["variant_id"],
                        },
                        LLMResponseError("invalid"),
                    ),
                ]
                run_batch(
                    manifest,
                    1,
                    1,
                    temp_dir,
                    "offline-test-model",
                    2,
                    12,
                    0.0,
                    completed_ids=set(),
                    total_items=len(manifest),
                    continue_on_retryable_failure=True,
                )
            progress = self._read_json(os.path.join(temp_dir, "progress.json"))
            from run_baseline_sweep import save_aggregate_summary

            aggregate = save_aggregate_summary(temp_dir, progress, manifest=manifest)

            self.assertEqual(aggregate["completed_episodes"], 1)
            self.assertEqual(aggregate["failed_episodes"], 1)
            self.assertEqual(aggregate["baseline_aggregates"][BASELINE_SINGLE_AGENT]["episodes"], 1)

    def test_retry_failures_requires_resume_and_run_dir(self):
        with self.assertRaises(ValueError):
            run_for_model(
                "offline-test-model",
                self._single_agent_config([200]),
                retry_failures=True,
            )

    def test_retry_selects_only_failed_items_and_preserves_original_error(self):
        cfg = self._single_agent_config([200, 201])
        original_run_single = run_single_episode
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "retry-run")

            def fail_second(*args, **kwargs):
                if args[1] == 201:
                    raise self._error_for_call(args, kwargs, LLMResponseError("invalid"))
                return original_run_single(*args, **kwargs)

            with patch("run_baseline_sweep.run_single_episode", side_effect=fail_second):
                with self.assertRaises(EpisodeCollectionError):
                    run_for_model("offline-test-model", cfg, run_dir=run_dir)

            original_error_files = sorted(os.listdir(os.path.join(run_dir, "errors")))
            retry_calls = []

            def record_retry(*args, **kwargs):
                retry_calls.append(kwargs["manifest_id"])
                return original_run_single(*args, **kwargs)

            with patch("run_baseline_sweep.run_single_episode", side_effect=record_retry):
                result = run_for_model(
                    "offline-test-model",
                    cfg,
                    run_dir=run_dir,
                    resume=True,
                    retry_failures=True,
                )

            manifest = build_episode_manifest([BASELINE_SINGLE_AGENT], episode_seeds=[200, 201])
            self.assertEqual(retry_calls, [manifest[1]["manifest_id"]])
            self.assertEqual(len(os.listdir(os.path.join(run_dir, "episodes"))), 2)
            self.assertEqual(sorted(os.listdir(os.path.join(run_dir, "errors"))), original_error_files)
            self.assertEqual(result["completed_episodes"], 2)
            self.assertEqual(result["failed_episodes"], 0)
            progress = self._read_json(os.path.join(run_dir, "progress.json"))
            self.assertTrue(all(entry["status"] == "completed" for entry in progress["items"].values()))

    def test_repeatedly_failed_retry_remains_failed_and_adds_audit_artifact(self):
        cfg = self._single_agent_config([200])
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "retry-run")
            original_error = LLMResponseError("invalid first response")

            def fail_first(*args, **kwargs):
                raise self._error_for_call(args, kwargs, original_error)

            with patch("run_baseline_sweep.run_single_episode", side_effect=fail_first):
                with self.assertRaises(EpisodeCollectionError):
                    run_for_model("offline-test-model", cfg, run_dir=run_dir)

            with patch("run_baseline_sweep.run_single_episode", side_effect=fail_first):
                result = run_for_model(
                    "offline-test-model",
                    cfg,
                    run_dir=run_dir,
                    resume=True,
                    retry_failures=True,
                )

            progress = self._read_json(os.path.join(run_dir, "progress.json"))
            self.assertEqual(next(iter(progress["items"].values()))["status"], "failed")
            self.assertEqual(len(os.listdir(os.path.join(run_dir, "errors"))), 2)
            self.assertEqual(result["completed_episodes"], 0)
            self.assertEqual(result["failed_episodes"], 1)

    def test_retry_with_no_failed_items_makes_no_episode_calls(self):
        cfg = self._single_agent_config([200])
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "complete-run")
            run_for_model("offline-test-model", cfg, run_dir=run_dir)
            with patch("run_baseline_sweep.run_single_episode") as mocked:
                result = run_for_model(
                    "offline-test-model",
                    cfg,
                    run_dir=run_dir,
                    resume=True,
                    retry_failures=True,
                )
            mocked.assert_not_called()
            self.assertEqual(result["failed_episodes"], 0)

    def test_missing_or_malformed_episode_artifact_stops_resume(self):
        cfg = self._single_agent_config([200])
        for mutation in ("missing", "malformed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                run_dir = os.path.join(temp_dir, "artifact-run")
                run_for_model("offline-test-model", cfg, run_dir=run_dir)
                episode_dir = next(
                    os.path.join(run_dir, "episodes", name)
                    for name in os.listdir(os.path.join(run_dir, "episodes"))
                )
                artifact_path = os.path.join(episode_dir, "episode.jsonl")
                if mutation == "missing":
                    os.remove(artifact_path)
                else:
                    with open(artifact_path, "w", encoding="utf-8") as handle:
                        handle.write("not-json\n")

                with patch("run_baseline_sweep.run_single_episode") as mocked:
                    with self.assertRaises(ArtifactValidationError):
                        run_for_model(
                            "offline-test-model",
                            cfg,
                            run_dir=run_dir,
                            resume=True,
                        )
                mocked.assert_not_called()

    def test_duplicate_manifest_items_and_completed_artifacts_stop_resume(self):
        cfg = self._single_agent_config([200])
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "duplicate-run")
            original_run_single = run_single_episode
            with patch("run_baseline_sweep.run_single_episode", wraps=original_run_single):
                run_for_model("offline-test-model", cfg, run_dir=run_dir)
            episode_names = os.listdir(os.path.join(run_dir, "episodes"))
            shutil.copytree(
                os.path.join(run_dir, "episodes", episode_names[0]),
                os.path.join(run_dir, "episodes", "duplicate-artifact"),
            )
            with self.assertRaises(RuntimeError):
                run_for_model("offline-test-model", cfg, run_dir=run_dir, resume=True)

        item = build_episode_manifest([BASELINE_SINGLE_AGENT], episode_seeds=[200])[0]
        duplicate_manifest = [item, dict(item)]
        progress = {"total_items": 2, "items": {}, "batches_completed": []}
        from run_baseline_sweep import _validate_progress_entries

        with self.assertRaises(ValueError):
            _validate_progress_entries(progress, duplicate_manifest)

    def test_frozen_configuration_mismatch_stops_before_episode_calls(self):
        cfg = self._single_agent_config([200])
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "frozen-run")
            run_for_model("offline-test-model", cfg, run_dir=run_dir)
            config_path = os.path.join(run_dir, "experiment_config.json")
            config = self._read_json(config_path)
            config["step_budget"] = 999
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            with patch("run_baseline_sweep.run_single_episode") as mocked:
                with self.assertRaises(ValueError):
                    run_for_model("offline-test-model", cfg, run_dir=run_dir, resume=True)
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()