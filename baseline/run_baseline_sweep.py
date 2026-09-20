from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set

from llm_client import (
    LLMRequestError,
)
from settings import SETTINGS, get_required, require_keys
from tool_use_baselines import (
    LLMCallError,
    LLMFallbackError,
    LLMResponseError,
    run_policy_episode,
    summarize_baseline_episode,
    VALID_BASELINES,
)
from tool_use_environment import (
    ENVIRONMENT_VERSION,
    SCHEMA_VERSION,
    build_bugfix_task_fixtures,
    persist_episode_run,
)


MANIFEST_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 2
MANIFEST_FIELDS = ("baseline", "repetition", "seed", "template_id", "variant_id")
COLLECTION_COMPLETED = "completed"
COLLECTION_FAILED = "failed"
RETRYABLE_EPISODE_FAILURE = "retryable_episode_failure"
FATAL_AUTHENTICATION = "authentication_failure"
FATAL_QUOTA = "quota_failure"
FATAL_ACCESS = "non_recoverable_access_failure"
FATAL_API = "repeated_api_failure"
FATAL_STRICT_FALLBACK = "strict_llm_fallback"
FATAL_ARTIFACT = "artifact_integrity_failure"
FATAL_MANIFEST = "manifest_integrity_failure"
FATAL_CONFIGURATION = "configuration_mismatch"
FATAL_FILESYSTEM = "filesystem_failure"
FATAL_DEPENDENCY = "dependency_failure"
FATAL_UNKNOWN = "unknown_failure"
VALID_COLLECTION_STATUSES = {COLLECTION_COMPLETED, COLLECTION_FAILED}


class ArtifactValidationError(ValueError):
    pass


def _exception_chain(error: Exception) -> List[Exception]:
    chain = []
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, Exception):
            chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _structured_failure_category(error: Exception) -> Optional[str]:
    for attribute in ("collection_error_category", "error_category", "fatal_category"):
        value = getattr(error, attribute, None)
        if isinstance(value, str):
            return value

    for attribute in ("policy_audit", "audit"):
        audit = getattr(error, attribute, None)
        if not isinstance(audit, dict):
            continue
        if audit.get("strict") is True and (
            audit.get("fallback_response_count", 0) or audit.get("fallback_action_count", 0)
        ):
            return FATAL_STRICT_FALLBACK
        if audit.get("fatal") is True:
            return FATAL_UNKNOWN
        for key in ("collection_error_category", "fatal_category", "error_category"):
            value = audit.get(key)
            if isinstance(value, str):
                return value
    return None


def _normalize_failure_category(category: str) -> str:
    if category in {
        FATAL_AUTHENTICATION,
        FATAL_QUOTA,
        FATAL_ACCESS,
        FATAL_API,
        FATAL_STRICT_FALLBACK,
        FATAL_ARTIFACT,
        FATAL_MANIFEST,
        FATAL_CONFIGURATION,
        FATAL_FILESYSTEM,
        FATAL_DEPENDENCY,
        FATAL_UNKNOWN,
    }:
        return category
    return {
        "authentication": FATAL_AUTHENTICATION,
        "quota": FATAL_QUOTA,
        "quota_or_rate_limit": FATAL_QUOTA,
        "non_recoverable_access": FATAL_ACCESS,
        "api_failure": FATAL_API,
        "repeated_api_failure": FATAL_API,
        "dependency": FATAL_DEPENDENCY,
        "configuration": FATAL_CONFIGURATION,
        "filesystem": FATAL_FILESYSTEM,
        "artifact": FATAL_ARTIFACT,
        "manifest": FATAL_MANIFEST,
        "strict_llm_fallback": FATAL_STRICT_FALLBACK,
        RETRYABLE_EPISODE_FAILURE: RETRYABLE_EPISODE_FAILURE,
    }.get(category, FATAL_UNKNOWN)


def _exception_status_code(error: Exception) -> Any:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return response.get("status_code")
    return getattr(response, "status_code", None)


def classify_episode_error(error: Exception) -> Dict[str, Any]:
    chain = _exception_chain(error)

    for candidate in chain:
        if isinstance(candidate, LLMFallbackError):
            return {
                "classification": FATAL_STRICT_FALLBACK,
                "retryable": False,
                "reason": "strict LLM fallback response or action was observed",
            }
        if isinstance(candidate, ArtifactValidationError):
            return {
                "classification": FATAL_ARTIFACT,
                "retryable": False,
                "reason": "episode artifact failed structural or identity validation",
            }

        structured = _structured_failure_category(candidate)
        if structured is not None:
            normalized = _normalize_failure_category(structured)
            if normalized != RETRYABLE_EPISODE_FAILURE:
                return {
                    "classification": normalized,
                    "retryable": False,
                    "reason": f"structured error category: {structured}",
                }

        if isinstance(candidate, LLMRequestError):
            normalized = _normalize_failure_category(candidate.error_category)
            return {
                "classification": normalized,
                "retryable": False,
                "reason": f"LLM request failed after {candidate.attempt_count} client attempts",
            }
        if isinstance(candidate, (PermissionError, OSError)):
            return {
                "classification": FATAL_FILESYSTEM,
                "retryable": False,
                "reason": f"filesystem error: {type(candidate).__name__}",
            }
        if isinstance(candidate, (ImportError, ModuleNotFoundError)):
            return {
                "classification": FATAL_DEPENDENCY,
                "retryable": False,
                "reason": f"dependency error: {type(candidate).__name__}",
            }

        exception_name = type(candidate).__name__
        if exception_name in {"AuthenticationError", "InvalidAPIKeyError"}:
            return {
                "classification": FATAL_AUTHENTICATION,
                "retryable": False,
                "reason": f"structured API exception class: {exception_name}",
            }
        if exception_name in {"QuotaExceededError", "InsufficientQuotaError"}:
            return {
                "classification": FATAL_QUOTA,
                "retryable": False,
                "reason": f"structured API exception class: {exception_name}",
            }
        if exception_name in {"PermissionDeniedError", "AuthorizationError"}:
            return {
                "classification": FATAL_ACCESS,
                "retryable": False,
                "reason": f"structured API exception class: {exception_name}",
            }

        status_code = _exception_status_code(candidate)
        if isinstance(status_code, int):
            if status_code == 401:
                return {
                    "classification": FATAL_AUTHENTICATION,
                    "retryable": False,
                    "reason": "structured API status code 401",
                }
            if status_code == 403:
                return {
                    "classification": FATAL_ACCESS,
                    "retryable": False,
                    "reason": "structured API status code 403",
                }
            if status_code == 429:
                return {
                    "classification": FATAL_QUOTA,
                    "retryable": False,
                    "reason": "structured API status code 429",
                }
            if 400 <= status_code < 500:
                return {
                    "classification": FATAL_ACCESS,
                    "retryable": False,
                    "reason": f"structured API status code {status_code}",
                }
            if status_code >= 500:
                return {
                    "classification": FATAL_API,
                    "retryable": False,
                    "reason": f"structured API status code {status_code}",
                }

    for candidate in chain:
        if isinstance(candidate, LLMResponseError):
            return {
                "classification": RETRYABLE_EPISODE_FAILURE,
                "retryable": True,
                "reason": "final model response/action remained invalid after parser attempts",
            }

    for candidate in chain:
        if isinstance(candidate, LLMCallError):
            return {
                "classification": FATAL_UNKNOWN,
                "retryable": False,
                "reason": "LLM call failed without a recognized retry-exhaustion category",
            }

    return {
        "classification": FATAL_UNKNOWN,
        "retryable": False,
        "reason": f"unrecognized episode exception: {type(error).__name__}",
    }


def create_episode_batches(baselines, episode_seeds, batch_size=20):
    all_configs = []
    for baseline_name in baselines:
        for seed in episode_seeds:
            all_configs.append((baseline_name, seed))

    batches = []
    for index in range(0, len(all_configs), batch_size):
        batches.append(all_configs[index:index + batch_size])
    return batches


def _manifest_id(baseline_name: str, repetition: int, seed: int, template_id: str, variant_id: str) -> str:
    return (
        f"{baseline_name}__rep_{repetition:03d}__seed_{seed}__"
        f"{template_id}__{variant_id}"
    )


def _manifest_item(
    *,
    baseline_name: str,
    repetition: int,
    seed: int,
    template_id: str,
    variant_id: str,
) -> Dict[str, Any]:
    item = {
        "manifest_id": _manifest_id(baseline_name, repetition, seed, template_id, variant_id),
        "baseline": baseline_name,
        "repetition": repetition,
        "seed": seed,
        "template_id": template_id,
        "variant_id": variant_id,
    }
    return item


def validate_manifest(
    manifest: Sequence[Dict[str, Any]],
    *,
    expected_baselines: Optional[Sequence[str]] = None,
    expected_fixtures: Optional[Sequence[Any]] = None,
    repetitions_per_fixture: Optional[int] = None,
) -> None:
    required_fields = set(MANIFEST_FIELDS).union({"manifest_id"})
    seen_ids: Set[str] = set()
    fixture_counts = defaultdict(int)
    baseline_counts = defaultdict(int)

    for item in manifest:
        if set(item) < required_fields:
            missing = sorted(required_fields - set(item))
            raise ValueError(f"Manifest item is missing fields: {missing}")
        if item["manifest_id"] in seen_ids:
            raise ValueError(f"Duplicate manifest item: {item['manifest_id']}")
        seen_ids.add(item["manifest_id"])
        expected_id = _manifest_id(
            item["baseline"],
            item["repetition"],
            item["seed"],
            item["template_id"],
            item["variant_id"],
        )
        if item["manifest_id"] != expected_id:
            raise ValueError(f"Manifest ID does not match item identity: {item['manifest_id']}")
        fixture_key = (item["template_id"], item["variant_id"])
        fixture_counts[fixture_key] += 1
        baseline_counts[item["baseline"]] += 1

    if expected_baselines is not None:
        expected_baseline_set = set(expected_baselines)
        if set(baseline_counts) != expected_baseline_set:
            raise ValueError(
                f"Manifest baselines {sorted(baseline_counts)} do not match "
                f"{sorted(expected_baseline_set)}"
            )
        if len(set(expected_baselines)) != len(expected_baselines):
            raise ValueError("Manifest baseline order cannot contain duplicates")
        baseline_episode_counts = set(baseline_counts.values())
        if len(baseline_episode_counts) > 1:
            raise ValueError("Manifest is not balanced across baselines")

    if expected_fixtures is not None:
        expected_fixture_keys = {
            (fixture.template_id, fixture.variant_id) for fixture in expected_fixtures
        }
        if set(fixture_counts) != expected_fixture_keys:
            raise ValueError("Manifest does not cover the complete fixture set")
        if repetitions_per_fixture is not None:
            expected_count = repetitions_per_fixture * len(expected_baselines or [])
            if set(fixture_counts.values()) != {expected_count}:
                raise ValueError("Manifest fixture repetitions are not balanced")


def build_episode_manifest(
    baselines: Sequence[str],
    *,
    fixtures: Optional[Sequence[Any]] = None,
    episode_seeds: Optional[Sequence[int]] = None,
    repetitions_per_fixture: int = 1,
    seed_start: int = 200,
) -> List[Dict[str, Any]]:
    if not baselines:
        raise ValueError("At least one baseline is required")
    unknown = sorted(set(baselines) - VALID_BASELINES)
    if unknown:
        raise ValueError(f"Unknown baselines in manifest: {unknown}")
    if len(set(baselines)) != len(baselines):
        raise ValueError("Manifest baselines cannot contain duplicates")

    fixture_set = list(fixtures or build_bugfix_task_fixtures())
    if not fixture_set:
        raise ValueError("At least one fixture is required")

    manifest: List[Dict[str, Any]] = []
    if episode_seeds is not None:
        if not episode_seeds:
            raise ValueError("episode_seeds must be non-empty")
        for repetition, seed in enumerate(episode_seeds, start=1):
            selected_fixture = fixture_set[random.Random(seed).randrange(len(fixture_set))]
            for baseline_name in baselines:
                manifest.append(
                    _manifest_item(
                        baseline_name=baseline_name,
                        repetition=repetition,
                        seed=seed,
                        template_id=selected_fixture.template_id,
                        variant_id=selected_fixture.variant_id,
                    )
                )
        validate_manifest(manifest, expected_baselines=baselines)
        return manifest

    if repetitions_per_fixture < 1:
        raise ValueError("repetitions_per_fixture must be at least 1")
    for fixture_index, fixture in enumerate(fixture_set):
        for repetition in range(1, repetitions_per_fixture + 1):
            seed = seed_start + fixture_index * repetitions_per_fixture + repetition - 1
            for baseline_name in baselines:
                manifest.append(
                    _manifest_item(
                        baseline_name=baseline_name,
                        repetition=repetition,
                        seed=seed,
                        template_id=fixture.template_id,
                        variant_id=fixture.variant_id,
                    )
                )

    validate_manifest(
        manifest,
        expected_baselines=baselines,
        expected_fixtures=fixture_set,
        repetitions_per_fixture=repetitions_per_fixture,
    )
    return manifest


def create_manifest_batches(manifest: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [list(manifest[index:index + batch_size]) for index in range(0, len(manifest), batch_size)]


def _progress_path(batch_folder: str) -> str:
    return os.path.join(batch_folder, "progress.json")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_progress(
    batch_folder: str,
    *,
    total_items: Optional[int] = None,
    total_batches: Optional[int] = None,
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    progress_file = _progress_path(batch_folder)
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        if not isinstance(progress, dict):
            raise ValueError("Progress file must contain a JSON object")
        if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise ValueError("Unsupported progress schema; refusing to resume this run")
        progress.setdefault("items", {})
        progress.setdefault("batches_completed", [])
        if not isinstance(progress["items"], dict) or not isinstance(progress["batches_completed"], list):
            raise ValueError("Progress items and batch checkpoints must have valid shapes")
        if total_items is not None and progress.get("total_items") != total_items:
            raise ValueError("Progress total does not match the frozen manifest")
        _refresh_progress_metadata(progress)
        return progress

    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "total_items": total_items,
        "total_batches": total_batches,
        "start_time": start_time if start_time is not None else time.time(),
        "batches_completed": [],
        "items": {},
    }
    _refresh_progress_metadata(progress)
    _write_json(progress_file, progress)
    return progress


def _refresh_progress_metadata(
    progress: Dict[str, Any],
    manifest: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    items = progress.get("items")
    if not isinstance(items, dict):
        raise ValueError("Progress items must be an object")

    completed_count = 0
    failed_ids = []
    for manifest_id, entry in items.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Progress entry is not an object: {manifest_id}")
        status = entry.get("status")
        if status not in VALID_COLLECTION_STATUSES:
            raise ValueError(f"Unsupported progress status for {manifest_id}: {status}")
        if status == COLLECTION_COMPLETED:
            completed_count += 1
        else:
            failed_ids.append(manifest_id)

    manifest_ids = [item["manifest_id"] for item in manifest] if manifest is not None else list(items)
    failed_manifest_ids = [manifest_id for manifest_id in manifest_ids if manifest_id in failed_ids]
    if manifest is None:
        failed_manifest_ids = failed_ids

    total_items = progress.get("total_items")
    if total_items is None:
        pending_count = None
        run_complete = False
    else:
        pending_count = total_items - completed_count - len(failed_ids)
        if pending_count < 0:
            raise ValueError("Progress contains more terminal items than the frozen manifest")
        run_complete = completed_count == total_items and not failed_ids

    progress["completed_count"] = completed_count
    progress["failed_count"] = len(failed_ids)
    progress["failed_manifest_ids"] = failed_manifest_ids
    progress["pending_count"] = pending_count
    progress["run_complete"] = run_complete
    progress["run_status"] = "complete" if run_complete else "partial"
    return progress


def _result_manifest_id(result: Dict[str, Any]) -> str:
    manifest_id = result.get("manifest_id")
    if manifest_id:
        return manifest_id
    return _manifest_id(
        result["baseline_name"],
        result.get("repetition", 1),
        result["seed"],
        result.get("template_id", "unknown_template"),
        result.get("variant_id", "unknown_variant"),
    )


def _store_progress_item(
    batch_folder: str,
    result: Dict[str, Any],
    *,
    total_items: Optional[int] = None,
) -> Dict[str, Any]:
    progress = _load_progress(batch_folder, total_items=total_items)
    manifest_id = _result_manifest_id(result)
    status = result.get("collection_status", "completed")
    if status not in VALID_COLLECTION_STATUSES:
        raise ValueError(f"Unsupported collection status for {manifest_id}: {status}")
    existing = progress["items"].get(manifest_id)
    if existing and existing.get("status") == "completed" and status == "completed":
        previous_run_dir = existing.get("result", {}).get("run_dir")
        current_run_dir = result.get("run_dir")
        if previous_run_dir and current_run_dir and previous_run_dir != current_run_dir:
            raise RuntimeError(f"Duplicate completed manifest item: {manifest_id}")
        return progress
    if existing and existing.get("status") == "completed":
        raise RuntimeError(f"Completed manifest item cannot be overwritten: {manifest_id}")

    progress["items"][manifest_id] = {
        "manifest_id": manifest_id,
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "result": result,
    }
    _refresh_progress_metadata(progress)
    _write_json(_progress_path(batch_folder), progress)
    return progress


def save_progress(
    batch_folder,
    batch_num,
    batch_results,
    total_batches,
    start_time,
    *,
    planned_manifest_ids=None,
    batch_complete=True,
    batch_status=None,
    batch_id="main",
):
    progress = _load_progress(
        batch_folder,
        total_batches=total_batches,
        start_time=start_time,
    )
    outcome_counts = defaultdict(int)
    unique_results = {}
    for result in batch_results:
        manifest_id = _result_manifest_id(result)
        if manifest_id in unique_results:
            raise RuntimeError(f"Duplicate batch result for manifest item: {manifest_id}")
        unique_results[manifest_id] = result
        _store_progress_item(batch_folder, result)

    progress = _load_progress(batch_folder)
    planned_ids = list(planned_manifest_ids) if planned_manifest_ids is not None else list(unique_results)
    batch_entries = [
        progress["items"][manifest_id]
        for manifest_id in planned_ids
        if manifest_id in progress["items"]
    ]
    batch_records = [entry["result"] for entry in batch_entries]
    completed_records = [
        result for entry, result in zip(batch_entries, batch_records)
        if entry.get("status") == COLLECTION_COMPLETED
    ]
    failed_manifest_ids = [
        entry["manifest_id"] for entry in batch_entries if entry.get("status") == COLLECTION_FAILED
    ]
    for result in completed_records:
        if "outcome" in result:
            outcome_counts[result["outcome"]] += 1

    completed_count = len(completed_records)
    failed_count = len(failed_manifest_ids)
    if batch_status is None:
        if not batch_complete:
            batch_status = "in_progress"
        elif failed_count:
            batch_status = "failed"
        elif completed_count == len(planned_ids):
            batch_status = "completed"
        else:
            batch_status = "partial"

    batch_record = {
        "batch_id": batch_id,
        "batch_num": batch_num,
        "planned_count": len(planned_ids),
        "episode_count": len(batch_records),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "failed_manifest_ids": failed_manifest_ids,
        "status": batch_status,
        "completed_at": datetime.now().isoformat(),
        "outcome_counts": dict(outcome_counts),
        "results": batch_records,
    }
    matching_batch = next(
        (
            index
            for index, batch in enumerate(progress["batches_completed"])
            if batch.get("batch_id", "main") == batch_id and batch.get("batch_num") == batch_num
        ),
        None,
    )
    if matching_batch is None:
        progress["batches_completed"].append(batch_record)
    else:
        progress["batches_completed"][matching_batch] = batch_record

    _refresh_progress_metadata(progress)
    _write_json(_progress_path(batch_folder), progress)
    return progress


def save_aggregate_summary(batch_folder, progress, *, manifest=None):
    _refresh_progress_metadata(progress, manifest)
    if manifest is not None:
        _write_json(_progress_path(batch_folder), progress)

    grouped = defaultdict(list)
    results = [
        entry["result"]
        for entry in progress.get("items", {}).values()
        if entry.get("status") == COLLECTION_COMPLETED
    ]

    for result in results:
        grouped[result["baseline_name"]].append(result)

    aggregate = {}
    for baseline_name, baseline_results in grouped.items():
        total = len(baseline_results)
        aggregate[baseline_name] = {
            "episodes": total,
            "success_rate": sum(1 for result in baseline_results if result["success"]) / total if total else 0.0,
            "partial_rate": sum(1 for result in baseline_results if result["partial"]) / total if total else 0.0,
            "failure_rate": sum(1 for result in baseline_results if result["failure"]) / total if total else 0.0,
            "mean_invalid_call_rate": sum(result["invalid_call_rate"] for result in baseline_results) / total if total else 0.0,
            "mean_calls_to_completion": sum(result["calls_to_completion"] for result in baseline_results) / total if total else 0.0,
            "mean_patch_acceptance_ratio": sum(result["patch_acceptance_ratio"] for result in baseline_results) / total if total else 0.0,
        }

    manifest_ids = [item["manifest_id"] for item in manifest] if manifest is not None else list(progress.get("items", {}))
    status_by_id = {
        manifest_id: entry.get("status")
        for manifest_id, entry in progress.get("items", {}).items()
    }
    failed_manifest_ids = [
        manifest_id for manifest_id in manifest_ids if status_by_id.get(manifest_id) == COLLECTION_FAILED
    ]
    pending_manifest_ids = [
        manifest_id for manifest_id in manifest_ids if manifest_id not in status_by_id
    ]
    planned_episodes = len(manifest_ids) if manifest is not None else progress.get("total_items")
    completed_episodes = len(results)
    failed_episodes = len(failed_manifest_ids)
    pending_episodes = len(pending_manifest_ids) if manifest is not None else progress.get("pending_count")
    run_complete = (
        planned_episodes is not None
        and completed_episodes == planned_episodes
        and failed_episodes == 0
        and pending_episodes == 0
    )
    summary = {
        "generated_at": datetime.now().isoformat(),
        "collection_status": "complete" if run_complete else "partial",
        "run_complete": run_complete,
        "planned_episodes": planned_episodes,
        "completed_episodes": completed_episodes,
        "failed_episodes": failed_episodes,
        "pending_episodes": pending_episodes,
        "failed_count": failed_episodes,
        "failed_manifest_ids": failed_manifest_ids,
        "pending_manifest_ids": pending_manifest_ids,
        "baseline_aggregates": aggregate,
    }
    _write_json(os.path.join(batch_folder, "aggregate_summary.json"), summary)
    return summary


def load_runner_settings(section="baseline_sweep"):
    require_keys(
        section,
        [
            "environment",
            "models",
            "baselines",
            "num_agents",
            "step_budget",
            "batch_size",
            "results_prefix",
        ],
    )

    models = get_required(section, "models")
    baselines = get_required(section, "baselines")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{section}.models must be a non-empty list")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError(f"{section}.baselines must be a non-empty list")
    unknown = sorted(set(baselines) - VALID_BASELINES)
    if unknown:
        raise ValueError(f"Unknown baselines in settings: {unknown}")

    episode_seeds = get_required(section, "episode_seeds") if "episode_seeds" in SETTINGS[section] else None
    fixture_repetitions = (
        get_required(section, "fixture_repetitions")
        if "fixture_repetitions" in SETTINGS[section]
        else None
    )
    if episode_seeds is None and fixture_repetitions is None:
        raise ValueError(f"{section} must define episode_seeds or fixture_repetitions")
    if episode_seeds is not None and fixture_repetitions is not None:
        raise ValueError(f"{section} cannot define both episode_seeds and fixture_repetitions")
    if episode_seeds is not None and (not isinstance(episode_seeds, list) or not episode_seeds):
        raise ValueError(f"{section}.episode_seeds must be a non-empty list")

    return {
        "environment": get_required(section, "environment"),
        "models": models,
        "baselines": baselines,
        "num_agents": get_required(section, "num_agents"),
        "step_budget": get_required(section, "step_budget"),
        "episode_seeds": episode_seeds,
        "fixture_repetitions": fixture_repetitions,
        "seed_start": SETTINGS[section].get("seed_start", 200),
        "batch_size": get_required(section, "batch_size"),
        "results_prefix": get_required(section, "results_prefix"),
        "strict_llm": bool(SETTINGS[section].get("strict_llm", False)),
    }


def _frozen_configuration(model_name: str, cfg: Dict[str, Any], manifest: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fixture_ids = []
    for item in manifest:
        fixture_id = (item["template_id"], item["variant_id"])
        if fixture_id not in fixture_ids:
            fixture_ids.append(fixture_id)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "environment": cfg["environment"],
        "model_name": model_name,
        "baselines": list(cfg["baselines"]),
        "num_agents": cfg["num_agents"],
        "step_budget": cfg["step_budget"],
        "strict_llm": bool(cfg.get("strict_llm", False)),
        "fixture_ids": [list(fixture_id) for fixture_id in fixture_ids],
        "fixture_repetitions": cfg["fixture_repetitions"],
        "episode_seeds": cfg["episode_seeds"],
        "seed_start": cfg["seed_start"],
        "batch_size": cfg["batch_size"],
        "total_episodes": len(manifest),
        "manifest_fields": list(MANIFEST_FIELDS),
        "created_at": datetime.now().isoformat(),
    }


def _build_manifest_for_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return build_episode_manifest(
        cfg["baselines"],
        fixtures=build_bugfix_task_fixtures(),
        episode_seeds=cfg.get("episode_seeds"),
        repetitions_per_fixture=cfg.get("fixture_repetitions") or 1,
        seed_start=cfg.get("seed_start", 200),
    )


def _prepare_batch_folder(model_name: str, cfg: Dict[str, Any], run_dir: Optional[str], resume: bool) -> str:
    if run_dir:
        batch_folder = os.path.normpath(run_dir)
        exists = os.path.exists(batch_folder)
        if resume and not exists:
            raise FileNotFoundError(f"Cannot resume missing run directory: {batch_folder}")
        if exists and not resume and os.listdir(batch_folder):
            raise FileExistsError(f"Run directory is not empty; use --resume: {batch_folder}")
        os.makedirs(batch_folder, exist_ok=True)
        return batch_folder

    if resume:
        raise ValueError("--resume requires --run-dir")
    model_safe_name = model_name.replace("/", "_").replace(":", "_")
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = os.path.join("results", f"{cfg['results_prefix']}_{model_safe_name}_{batch_timestamp}")
    if os.path.exists(batch_folder):
        raise FileExistsError(f"Run directory collision: {batch_folder}")
    os.makedirs(batch_folder)
    return batch_folder


def _load_or_write_run_state(
    batch_folder: str,
    model_name: str,
    cfg: Dict[str, Any],
    generated_manifest: Sequence[Dict[str, Any]],
    generated_config: Dict[str, Any],
    *,
    resume: bool,
) -> List[Dict[str, Any]]:
    manifest_path = os.path.join(batch_folder, "manifest.json")
    config_path = os.path.join(batch_folder, "experiment_config.json")
    if resume:
        if not os.path.exists(manifest_path) or not os.path.exists(config_path):
            raise FileNotFoundError("Resume requires manifest.json and experiment_config.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest_payload = json.load(handle)
        with open(config_path, "r", encoding="utf-8") as handle:
            stored_config = json.load(handle)
        if not isinstance(manifest_payload, dict) or manifest_payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Stored manifest schema differs from the frozen manifest")
        stored_manifest = manifest_payload.get("items")
        if not isinstance(stored_manifest, list) or not isinstance(stored_config, dict):
            raise ValueError("Stored manifest or configuration has an invalid shape")
        if stored_manifest != list(generated_manifest):
            raise ValueError("Stored manifest differs from the current deterministic plan")
        stored_comparable = {key: value for key, value in stored_config.items() if key != "created_at"}
        generated_comparable = {key: value for key, value in generated_config.items() if key != "created_at"}
        if stored_comparable != generated_comparable:
            mismatched_keys = sorted(set(stored_comparable) ^ set(generated_comparable))
            mismatched_keys.extend(
                key
                for key in sorted(set(stored_comparable).intersection(generated_comparable))
                if stored_comparable[key] != generated_comparable[key]
            )
            raise ValueError(f"Frozen configuration mismatch for {mismatched_keys[0]}")
        validate_manifest(stored_manifest, expected_baselines=cfg["baselines"])
        return stored_manifest

    if os.path.exists(manifest_path) or os.path.exists(config_path):
        raise FileExistsError("Run state already exists; refusing to overwrite it")
    _write_json(
        manifest_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "items": list(generated_manifest),
        },
    )
    _write_json(config_path, generated_config)
    return list(generated_manifest)


def _completed_artifacts(
    batch_folder: str,
    manifest: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    model_name: Optional[str] = None,
    num_agents: Optional[int] = None,
    step_budget: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    episodes_folder = os.path.join(batch_folder, "episodes")
    if not os.path.isdir(episodes_folder):
        return {}
    manifest_by_id = {item["manifest_id"]: item for item in manifest or []}
    completed: Dict[str, Dict[str, Any]] = {}
    for entry_name in sorted(os.listdir(episodes_folder)):
        run_dir = os.path.join(episodes_folder, entry_name)
        if not os.path.isdir(run_dir):
            raise ArtifactValidationError(f"Unexpected file in episodes directory: {run_dir}")
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise ArtifactValidationError(f"Completed episode is missing its summary artifact: {run_dir}")
        summary = _read_json_object(summary_path, "episode summary")
        if summary.get("collection_status") != "completed":
            raise ArtifactValidationError(f"Episode artifact is not marked completed: {summary_path}")
        manifest_id = summary.get("manifest_id")
        if not manifest_id:
            raise ArtifactValidationError(f"Completed episode is missing manifest identity: {summary_path}")
        if manifest_id in completed:
            raise RuntimeError(f"Duplicate completed manifest item: {manifest_id}")
        item = manifest_by_id.get(manifest_id) if manifest is not None else None
        if manifest is not None and item is None:
            raise ValueError(f"Artifact identity is not present in the frozen manifest: {manifest_id}")
        summary = _validate_completed_artifact(
            run_dir,
            item,
            model_name=model_name,
            num_agents=num_agents,
            step_budget=step_budget,
        )
        summary["run_dir"] = run_dir
        completed[manifest_id] = summary
    return completed


def _sync_completed_items(
    batch_folder: str,
    manifest: Sequence[Dict[str, Any]],
    *,
    model_name: Optional[str] = None,
    num_agents: Optional[int] = None,
    step_budget: Optional[int] = None,
) -> Set[str]:
    manifest_by_id = {item["manifest_id"]: item for item in manifest}
    progress = _load_progress(batch_folder, total_items=len(manifest))
    _validate_progress_entries(progress, manifest)
    artifact_results = _completed_artifacts(
        batch_folder,
        manifest,
        model_name=model_name,
        num_agents=num_agents,
        step_budget=step_budget,
    )
    for manifest_id, result in artifact_results.items():
        if manifest_id not in manifest_by_id:
            raise ValueError(f"Artifact identity is not present in the frozen manifest: {manifest_id}")
        existing = progress["items"].get(manifest_id)
        if existing and existing.get("status") == "completed":
            previous_run_dir = existing.get("result", {}).get("run_dir")
            if previous_run_dir and previous_run_dir != result["run_dir"]:
                raise RuntimeError(f"Duplicate completed manifest item: {manifest_id}")
            continue
        progress["items"][manifest_id] = {
            "manifest_id": manifest_id,
            "status": "completed",
            "updated_at": datetime.now().isoformat(),
            "result": result,
        }
    for manifest_id, entry in progress["items"].items():
        if entry.get("status") != "completed":
            continue
        result = entry.get("result", {})
        run_dir = result.get("run_dir")
        if not run_dir:
            raise ArtifactValidationError(f"Completed manifest item has no episode artifact: {manifest_id}")
        _validate_completed_artifact(
            run_dir,
            manifest_by_id[manifest_id],
            model_name=model_name,
            num_agents=num_agents,
            step_budget=step_budget,
        )

    _refresh_progress_metadata(progress, manifest)
    _write_json(_progress_path(batch_folder), progress)
    return {
        manifest_id
        for manifest_id, entry in progress["items"].items()
        if entry.get("status") == "completed"
    }


def _select_manifest_items(
    manifest: Sequence[Dict[str, Any]],
    *,
    progress: Optional[Dict[str, Any]] = None,
    retry_failures: bool = False,
    max_episodes: Optional[int] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if retry_failures:
        if progress is None:
            raise ValueError("Retry selection requires loaded progress")
        selected = [
            item
            for item in manifest
            if progress.get("items", {}).get(item["manifest_id"], {}).get("status") == COLLECTION_FAILED
        ]
    else:
        selected = list(manifest)
    if max_episodes is not None:
        counts = defaultdict(int)
        limited = []
        for item in selected:
            if counts[item["baseline"]] >= max_episodes:
                continue
            counts[item["baseline"]] += 1
            limited.append(item)
        selected = limited
    if max_items is not None:
        selected = selected[:max_items]
    return selected


def _expected_episode_num_agents(baseline_name: str, num_agents: int) -> int:
    if baseline_name in {"single_agent", "llm_single_agent"}:
        return 1
    return max(num_agents, 2)


def _read_json_object(path: str, description: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Malformed {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{description} must contain a JSON object: {path}")
    return payload


def _validate_error_artifact(error_path: str, manifest_id: str) -> None:
    if not os.path.isfile(error_path):
        raise ArtifactValidationError(f"Missing error artifact for failed manifest item: {manifest_id}")
    payload = _read_json_object(error_path, "episode error artifact")
    if payload.get("record_type") != "episode_error":
        raise ArtifactValidationError(f"Invalid error artifact record type: {error_path}")
    if payload.get("manifest_id") != manifest_id:
        raise ArtifactValidationError(f"Error artifact identity mismatch: {error_path}")


def _validate_completed_artifact(
    run_dir: str,
    item: Optional[Dict[str, Any]] = None,
    *,
    model_name: Optional[str] = None,
    num_agents: Optional[int] = None,
    step_budget: Optional[int] = None,
) -> Dict[str, Any]:
    if not os.path.isdir(run_dir):
        raise ArtifactValidationError(f"Missing completed episode directory: {run_dir}")

    summary_path = os.path.join(run_dir, "summary.json")
    episode_path = os.path.join(run_dir, "episode.jsonl")
    fixture_path = os.path.join(run_dir, "task_fixture.json")
    for path, description in (
        (summary_path, "summary artifact"),
        (episode_path, "episode artifact"),
        (fixture_path, "task fixture artifact"),
    ):
        if not os.path.isfile(path):
            raise ArtifactValidationError(f"Missing {description}: {path}")

    summary = _read_json_object(summary_path, "episode summary")
    if summary.get("collection_status") != COLLECTION_COMPLETED:
        raise ArtifactValidationError(f"Episode summary is not completed: {summary_path}")
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("environment_version") != ENVIRONMENT_VERSION:
        raise ArtifactValidationError(f"Episode summary schema mismatch: {summary_path}")

    expected_identity = {
        "manifest_id": item["manifest_id"] if item else summary.get("manifest_id"),
        "baseline_name": item["baseline"] if item else summary.get("baseline_name"),
        "repetition": item["repetition"] if item else summary.get("repetition"),
        "seed": item["seed"] if item else summary.get("seed"),
        "template_id": item["template_id"] if item else summary.get("template_id"),
        "variant_id": item["variant_id"] if item else summary.get("variant_id"),
    }
    if any(summary.get(key) != expected for key, expected in expected_identity.items()):
        raise ArtifactValidationError(f"Episode summary identity mismatch: {summary_path}")
    if model_name is not None and summary.get("model_name") != model_name:
        raise ArtifactValidationError(f"Episode summary model mismatch: {summary_path}")

    outcome = summary.get("outcome")
    if outcome not in {"success", "partial", "failure"}:
        raise ArtifactValidationError(f"Episode summary has invalid outcome: {summary_path}")
    if any(summary.get(flag) is not (outcome == flag) for flag in ("success", "partial", "failure")):
        raise ArtifactValidationError(f"Episode outcome flags are inconsistent: {summary_path}")

    policy_audit = summary.get("policy_audit", {})
    if not isinstance(policy_audit, dict):
        raise ArtifactValidationError(f"Episode policy audit is malformed: {summary_path}")
    if policy_audit.get("strict") is True and (
        policy_audit.get("fallback_response_count", 0) or policy_audit.get("fallback_action_count", 0)
    ):
        raise ArtifactValidationError(f"Strict episode contains an LLM fallback: {summary_path}")

    replay_spec = summary.get("replay_spec")
    if not isinstance(replay_spec, dict):
        raise ArtifactValidationError(f"Episode replay specification is missing: {summary_path}")
    replay_keys = ("seed", "template_id", "variant_id", "step_budget", "num_agents", "environment_version", "config_hash")
    if any(key not in replay_spec for key in replay_keys):
        raise ArtifactValidationError(f"Episode replay specification is incomplete: {summary_path}")
    for key in ("seed", "template_id", "variant_id", "environment_version", "config_hash"):
        if replay_spec[key] != summary.get(key):
            raise ArtifactValidationError(f"Episode replay specification mismatch: {summary_path}")
    if num_agents is not None and replay_spec["num_agents"] != _expected_episode_num_agents(
        summary["baseline_name"], num_agents
    ):
        raise ArtifactValidationError(f"Episode agent-count mismatch: {summary_path}")
    if step_budget is not None and replay_spec["step_budget"] != step_budget:
        raise ArtifactValidationError(f"Episode step-budget mismatch: {summary_path}")

    try:
        with open(episode_path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    except OSError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Malformed episode event artifact: {episode_path}") from exc
    if not records or not isinstance(records[0], dict) or records[0].get("record_type") != "episode_header":
        raise ArtifactValidationError(f"Episode header is missing or malformed: {episode_path}")
    header = records[0]
    for key, expected in {
        "schema_version": summary["schema_version"],
        "environment_version": replay_spec["environment_version"],
        "config_hash": replay_spec["config_hash"],
        "seed": replay_spec["seed"],
        "template_id": replay_spec["template_id"],
        "variant_id": replay_spec["variant_id"],
        "num_agents": replay_spec["num_agents"],
        "step_budget": replay_spec["step_budget"],
    }.items():
        if header.get(key) != expected:
            raise ArtifactValidationError(f"Episode header mismatch: {episode_path}")
    events = records[1:]
    if any(not isinstance(event, dict) or event.get("record_type") != "tool_event" for event in events):
        raise ArtifactValidationError(f"Episode event record is malformed: {episode_path}")
    if summary.get("event_count") != len(events):
        raise ArtifactValidationError(f"Episode event count mismatch: {episode_path}")

    fixture = _read_json_object(fixture_path, "task fixture")
    if (fixture.get("template_id"), fixture.get("variant_id")) != (
        summary["template_id"],
        summary["variant_id"],
    ):
        raise ArtifactValidationError(f"Task fixture identity mismatch: {fixture_path}")
    return summary


def _validate_progress_entries(
    progress: Dict[str, Any],
    manifest: Sequence[Dict[str, Any]],
) -> None:
    manifest_by_id = {item["manifest_id"]: item for item in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("Frozen manifest contains duplicate manifest IDs")
    if progress.get("total_items") != len(manifest):
        raise ValueError("Progress total does not match the frozen manifest")

    for manifest_id, entry in progress.get("items", {}).items():
        if manifest_id not in manifest_by_id:
            raise ValueError(f"Progress identity is not present in the frozen manifest: {manifest_id}")
        if not isinstance(entry, dict) or entry.get("manifest_id") != manifest_id:
            raise ValueError(f"Progress entry identity mismatch: {manifest_id}")
        status = entry.get("status")
        result = entry.get("result")
        if status not in VALID_COLLECTION_STATUSES or not isinstance(result, dict):
            raise ValueError(f"Invalid progress entry for manifest item: {manifest_id}")
        if result.get("manifest_id") != manifest_id:
            raise ValueError(f"Progress result identity mismatch: {manifest_id}")
        expected_result_values = {
            "baseline_name": manifest_by_id[manifest_id]["baseline"],
            "repetition": manifest_by_id[manifest_id]["repetition"],
            "seed": manifest_by_id[manifest_id]["seed"],
            "template_id": manifest_by_id[manifest_id]["template_id"],
            "variant_id": manifest_by_id[manifest_id]["variant_id"],
        }
        for key, expected in expected_result_values.items():
            if result.get(key) != expected:
                raise ValueError(f"Progress result identity mismatch for {manifest_id}: {key}")
        if status == COLLECTION_FAILED:
            error_path = result.get("error_path")
            if not isinstance(error_path, str):
                raise ArtifactValidationError(f"Failed progress entry has no error artifact: {manifest_id}")
            _validate_error_artifact(error_path, manifest_id)

    _refresh_progress_metadata(progress, manifest)


class EpisodeCollectionError(RuntimeError):
    def __init__(self, item: Dict[str, Any], error_path: str, original: Exception):
        self.item = item
        self.error_path = error_path
        self.original = original
        self.classification = classify_episode_error(original)
        super().__init__(
            f"Episode {item['manifest_id']} failed; persisted error at {error_path}: "
            f"classification={self.classification['classification']} "
            f"({self.classification['reason']}); {type(original).__name__}: {original}"
        )


def _persist_episode_error(batch_folder: str, item: Dict[str, Any], error: Exception) -> str:
    errors_folder = os.path.join(batch_folder, "errors")
    os.makedirs(errors_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    classification = classify_episode_error(error)
    error_path = os.path.join(errors_folder, f"{item['manifest_id']}_{timestamp}.json")
    payload = {
        "record_type": "episode_error",
        "manifest_id": item["manifest_id"],
        "baseline": item["baseline"],
        "repetition": item["repetition"],
        "seed": item["seed"],
        "template_id": item["template_id"],
        "variant_id": item["variant_id"],
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
        "classification": classification["classification"],
        "classification_reason": classification["reason"],
        "retryable": classification["retryable"],
    }
    if hasattr(error, "reason"):
        payload["fallback_reason"] = getattr(error, "reason")
    if hasattr(error, "policy_audit"):
        payload["policy_audit"] = getattr(error, "policy_audit")
    if isinstance(error, LLMRequestError):
        payload["client_attempt_count"] = error.attempt_count
        payload["client_retry_exhausted"] = error.retry_exhausted
        payload["client_error_category"] = error.error_category
    _write_json(error_path, payload)
    return error_path


def run_single_episode(
    baseline_name,
    seed,
    batch_folder,
    model_name,
    num_agents,
    step_budget,
    *,
    template_id=None,
    variant_id=None,
    repetition=1,
    manifest_id=None,
    strict_llm=False,
    llm_call=None,
):
    item = {
        "manifest_id": manifest_id or _manifest_id(
            baseline_name,
            repetition,
            seed,
            template_id or "implicit_template",
            variant_id or "implicit_variant",
        ),
        "baseline": baseline_name,
        "repetition": repetition,
        "seed": seed,
        "template_id": template_id,
        "variant_id": variant_id,
    }
    print(
        f"    Running baseline={baseline_name} repetition={repetition} seed={seed} "
        f"fixture={template_id}/{variant_id}"
    )
    try:
        environment, events = run_policy_episode(
            baseline_name=baseline_name,
            seed=seed,
            step_budget=step_budget,
            num_agents=num_agents,
            template_id=template_id,
            variant_id=variant_id,
            model_name=model_name,
            strict_llm=strict_llm,
            llm_call=llm_call,
        )
        summary = summarize_baseline_episode(
            baseline_name=baseline_name,
            environment=environment,
            events=events,
            model_name=model_name,
        )
        summary.update(
            {
                "collection_status": "completed",
                "manifest_id": item["manifest_id"],
                "repetition": repetition,
                "seed": seed,
                "template_id": template_id,
                "variant_id": variant_id,
            }
        )
        run_dir = persist_episode_run(
            output_root=os.path.join(batch_folder, "episodes"),
            environment=environment,
            events=events,
            run_label=item["manifest_id"],
            summary=summary,
        )
        _validate_completed_artifact(
            run_dir,
            item,
            model_name=model_name,
            num_agents=num_agents,
            step_budget=step_budget,
        )
        summary["run_dir"] = run_dir
        return summary
    except Exception as exc:
        error_path = _persist_episode_error(batch_folder, item, exc)
        raise EpisodeCollectionError(item, error_path, exc) from exc


def run_batch(
    batch_configs,
    batch_num,
    total_batches,
    batch_folder,
    model_name,
    num_agents,
    step_budget,
    start_time,
    *,
    strict_llm=False,
    completed_ids=None,
    total_items=None,
    continue_on_retryable_failure=False,
    batch_id="main",
):
    batch_results = []
    completed_ids = completed_ids if completed_ids is not None else set()
    planned_manifest_ids = [item["manifest_id"] for item in batch_configs]
    batch_start = time.time()
    print(f"\nBatch {batch_num}/{total_batches} with {len(batch_configs)} planned episodes")
    for item in batch_configs:
        if item["manifest_id"] in completed_ids:
            print(f"    Skipping completed manifest item {item['manifest_id']}")
            continue
        try:
            result = run_single_episode(
                item["baseline"],
                item["seed"],
                batch_folder,
                model_name,
                num_agents,
                step_budget,
                template_id=item["template_id"],
                variant_id=item["variant_id"],
                repetition=item["repetition"],
                manifest_id=item["manifest_id"],
                strict_llm=strict_llm,
            )
        except Exception as original_error:
            if isinstance(original_error, EpisodeCollectionError):
                exc = original_error
            else:
                error_path = _persist_episode_error(batch_folder, item, original_error)
                exc = EpisodeCollectionError(item, error_path, original_error)
            failure = {
                **item,
                "baseline_name": item["baseline"],
                "collection_status": COLLECTION_FAILED,
                "error_path": exc.error_path,
                "error_type": type(exc.original).__name__,
                "error_message": str(exc.original),
                "classification": exc.classification["classification"],
                "classification_reason": exc.classification["reason"],
                "retryable": exc.classification["retryable"],
            }
            batch_results.append(failure)
            _store_progress_item(batch_folder, failure, total_items=total_items)
            can_continue = continue_on_retryable_failure and exc.classification["retryable"]
            save_progress(
                batch_folder,
                batch_num,
                batch_results,
                total_batches,
                start_time,
                planned_manifest_ids=planned_manifest_ids,
                batch_complete=False,
                batch_status="in_progress" if can_continue else "aborted",
                batch_id=batch_id,
            )
            if can_continue:
                print(
                    f"    Isolated episode failure classified as retryable; continuing: "
                    f"{exc.classification['reason']}"
                )
                continue
            raise
        _store_progress_item(batch_folder, result, total_items=total_items)
        completed_ids.add(item["manifest_id"])
        batch_results.append(result)

    progress = save_progress(
        batch_folder,
        batch_num,
        batch_results,
        total_batches,
        start_time,
        planned_manifest_ids=planned_manifest_ids,
        batch_complete=True,
        batch_id=batch_id,
    )
    print(f"  Completed in {time.time() - batch_start:.2f}s")
    return progress


def run_for_model(
    model_name,
    cfg,
    *,
    run_dir=None,
    resume=False,
    max_episodes=None,
    max_items=None,
    continue_on_retryable_failure=False,
    retry_failures=False,
):
    if retry_failures and (not resume or not run_dir):
        raise ValueError("--retry-failures requires --resume and --run-dir")
    if cfg["environment"] != "synthetic_bugfix":
        raise ValueError(f"Unsupported batch environment: {cfg['environment']}")

    generated_manifest = _build_manifest_for_config(cfg)
    generated_config = _frozen_configuration(model_name, cfg, generated_manifest)
    batch_folder = _prepare_batch_folder(model_name, cfg, run_dir, resume)
    manifest = _load_or_write_run_state(
        batch_folder,
        model_name,
        cfg,
        generated_manifest,
        generated_config,
        resume=resume,
    )
    start_time = time.time()
    _load_progress(
        batch_folder,
        total_items=len(manifest),
        total_batches=None,
        start_time=start_time,
    )
    completed_ids = _sync_completed_items(
        batch_folder,
        manifest,
        model_name=model_name,
        num_agents=cfg["num_agents"],
        step_budget=cfg["step_budget"],
    )
    progress = _load_progress(batch_folder, total_items=len(manifest))
    _refresh_progress_metadata(progress, manifest)
    _write_json(_progress_path(batch_folder), progress)
    selected_manifest = _select_manifest_items(
        manifest,
        progress=progress,
        retry_failures=retry_failures,
        max_episodes=max_episodes,
        max_items=max_items,
    )
    batches = create_manifest_batches(selected_manifest, cfg["batch_size"])
    total_batches = len(batches)

    print(f"Running tool-use baseline sweep for model metadata {model_name}")
    print(f"Results folder: {batch_folder}")
    print(f"Frozen manifest: {os.path.join(batch_folder, 'manifest.json')}")

    if retry_failures and not selected_manifest:
        print("No failed manifest items exist to retry.")
        aggregate = save_aggregate_summary(batch_folder, progress, manifest=manifest)
        print(f"Aggregate summary written to: {os.path.join(batch_folder, 'aggregate_summary.json')}")
        return aggregate

    try:
        for batch_num, batch_configs in enumerate(batches, start=1):
            run_batch(
                batch_configs,
                batch_num,
                total_batches,
                batch_folder,
                model_name,
                cfg["num_agents"],
                cfg["step_budget"],
                start_time,
                strict_llm=cfg.get("strict_llm", False),
                completed_ids=completed_ids,
                total_items=len(manifest),
                continue_on_retryable_failure=continue_on_retryable_failure or retry_failures,
                batch_id="retry" if retry_failures else "main",
            )
    except EpisodeCollectionError:
        with open(_progress_path(batch_folder), "r", encoding="utf-8") as handle:
            progress = json.load(handle)
        save_aggregate_summary(batch_folder, progress, manifest=manifest)
        raise

    with open(_progress_path(batch_folder), "r", encoding="utf-8") as handle:
        progress = json.load(handle)
    aggregate = save_aggregate_summary(batch_folder, progress, manifest=manifest)

    completed_count = sum(
        1 for entry in progress.get("items", {}).values() if entry.get("status") == COLLECTION_COMPLETED
    )
    failed_count = sum(
        1 for entry in progress.get("items", {}).values() if entry.get("status") == COLLECTION_FAILED
    )
    print(f"Completed {completed_count} of {len(manifest)} planned episodes; failed={failed_count}")
    print(f"Aggregate summary written to: {os.path.join(batch_folder, 'aggregate_summary.json')}")
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run configured tool-use baseline sweeps.")
    parser.add_argument(
        "--settings-section",
        default="baseline_sweep",
        help="Settings section to run, such as baseline_sweep, llm_baseline_pilot, or thesis_experiment.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit executed episodes per baseline while retaining the full frozen manifest.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit executed manifest items globally while retaining the full frozen manifest.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory to create or resume instead of creating a new timestamped directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory after validating its frozen manifest and configuration.",
    )
    parser.add_argument(
        "--continue-on-retryable-failure",
        action="store_true",
        help="Continue collection after an isolated retryable episode failure.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="On resume, run only manifest items currently marked failed.",
    )
    args = parser.parse_args(argv)

    for argument_name, argument_value in (
        ("--max-episodes", args.max_episodes),
        ("--max-items", args.max_items),
    ):
        if argument_value is not None and argument_value < 1:
            raise ValueError(f"{argument_name} must be at least 1")
    if args.resume and not args.run_dir:
        raise ValueError("--resume requires --run-dir")
    if args.retry_failures and (not args.resume or not args.run_dir):
        raise ValueError("--retry-failures requires --resume and --run-dir")

    cfg = load_runner_settings(args.settings_section)
    if args.run_dir and len(cfg["models"]) != 1:
        raise ValueError("--run-dir requires exactly one configured model")

    print(f"Loaded {args.settings_section} settings from settings.py")
    for model_name in cfg["models"]:
        run_for_model(
            model_name,
            cfg,
            run_dir=args.run_dir,
            resume=args.resume,
            max_episodes=args.max_episodes,
            max_items=args.max_items,
            continue_on_retryable_failure=args.continue_on_retryable_failure,
            retry_failures=args.retry_failures,
        )


if __name__ == "__main__":
    total_start_time = time.time()
    try:
        main()
    finally:
        total_elapsed = time.time() - total_start_time
        print(f"Total program execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")