from typing import Any, Dict, Iterable


# Centralized runtime configuration for the experiment scripts.
# These comments preserve the intent of the earlier inline defaults.
SETTINGS: Dict[str, Any] = {
    # Large multi-model sweep configuration used by run_experiment_multi_model.py.
    "run_experiment_multi_model": {
        # Primary controlled environment used by the batch runner.
        "environment": "synthetic_bugfix",
        # Models retained as metadata so later LLM-backed baselines can share the same runner.
        "models": ["azureai/gpt-5-mini"],
        # Baseline policies included in the pilot.
        "baselines": [
            "single_agent",
            "random_policy",
            "independent_multi_agent",
            "prompted_coordination",
        ],
        # Primary multi-agent setting. Single-agent runs override this to one agent.
        "num_agents": 2,
        # Step budget for each episode.
        "step_budget": 12,
        # Deterministic pilot seeds. Expand this list for the full 200-400 episode pilot.
        "episode_seeds": list(range(200, 220)),
        # Number of episodes processed in one progress batch.
        "batch_size": 20,
        # Prefix used for generated result directories and files.
        "results_prefix": "tool_use_pilot",
    },
    # Minimal single-run example configuration.
    "single_run_experiment": {
        # Primary controlled environment used by the main entrypoint.
        "environment": "synthetic_bugfix",
        # Number of agents in the pilot single-run demo.
        "num_agents": 2,
        # Deterministic seed for fixture selection and replay.
        "seed": 17,
        # Episode-level step budget.
        "step_budget": 12,
        # Root folder for canonical event logs.
        "results_root": "results",
    },
    # Controlled tool-use environment contract.
    "tool_use_environment": {
        # Primary task family for the pilot.
        "task_family": "synthetic_bugfix",
        # Fixed-fixture pilot range. Generated variants remain deferred.
        "template_count": 8,
        "variants_per_template": 2,
        # Primary outcome variable.
        "outcome_space": ["success", "partial", "failure"],
        # Externally visible environment states.
        "progress_states": [
            "not_started",
            "in_progress",
            "completed_success",
            "completed_partial",
            "completed_failure",
        ],
        # Minimal tool set used in the pilot.
        "allowed_tools": [
            "retrieve_file",
            "search_symbol",
            "run_tests",
            "apply_patch",
            "finalize_ticket",
        ],
        # Invalid-call taxonomy used for reporting.
        "invalid_reason_classes": [
            "schema_invalid",
            "reference_invalid",
            "state_invalid",
            "semantic_noop",
            "budget_invalid",
        ],
    },
    # Count-based categorical MI estimator configuration for the pilot.
    "coordination_metrics": {
        # Primary estimator used for the first pilot runs.
        "estimator": "plugin",
        # Sensitivity analysis over Laplace / Dirichlet-style smoothing.
        "smoothing_alphas": [0.0, 0.5, 1.0],
        # Bootstrap iterations for confidence intervals.
        "bootstrap_iterations": 500,
        # Permutation-null iterations for significance checks.
        "permutation_iterations": 250,
        # Confidence level for bootstrap intervals.
        "confidence_level": 0.95,
    },
}


class SettingsError(ValueError):
    pass


def get_optional(section: str, key: str, default: Any = None) -> Any:
    section_data = SETTINGS.get(section, {})
    return section_data.get(key, default)


def get_required(section: str, key: str) -> Any:
    section_data = SETTINGS.get(section)
    if not isinstance(section_data, dict):
        raise SettingsError(f"Missing settings section: {section}")
    if key not in section_data:
        raise SettingsError(f"Missing required setting: {section}.{key}")
    value = section_data[key]
    if value is None:
        raise SettingsError(f"Required setting cannot be None: {section}.{key}")
    return value


def require_keys(section: str, keys: Iterable[str]) -> None:
    for key in keys:
        get_required(section, key)
