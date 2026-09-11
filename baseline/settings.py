from typing import Any, Dict, Iterable


# Centralized runtime configuration for the experiment scripts.
# These comments preserve the intent of the earlier inline defaults.
SETTINGS: Dict[str, Any] = {
    # Large baseline sweep configuration used by run_baseline_sweep.py.
    "baseline_sweep": {
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
        # Eight steps leave standard two-defect tasks solvable, but expose the
        # extra verification/update work in hard three-defect variants.
        "step_budget": 8,
        # Deterministic pilot seeds. Expand this list for the full 200-400 episode pilot.
        "episode_seeds": list(range(200, 220)),
        # Number of episodes processed in one progress batch.
        "batch_size": 20,
        # Prefix used for generated result directories and files.
        "results_prefix": "tool_use_pilot",
    },
    # Small opt-in configuration for real API-backed baseline validation.
    "llm_baseline_pilot": {
        "environment": "synthetic_bugfix",
        "models": ["azureai/gpt-5-mini"],
        "baselines": [
            "llm_single_agent",
            "llm_independent_multi_agent",
            "llm_prompted_coordination",
        ],
        "num_agents": 2,
        "step_budget": 12,
        "episode_seeds": list(range(200, 205)),
        "batch_size": 1,
        "results_prefix": "tool_use_llm_pilot",
    },
    # Defaults used by llm_client.py. Environment variables can override these values.
    "llm": {
        "default_model": "azureai/gpt-5-mini",
        "azure_openai_api_version": "2025-01-01-preview",
        "default_timeout_seconds": 45.0,
        "default_max_retries": 5,
        "default_base_delay_seconds": 1.0,
        "max_completion_tokens": 256,
        "action_parse_attempts": 2,
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
        # Fixed-fixture pilot range, including a small hard-variant slice.
        "template_count": 8,
        "variants_per_template": 2,
        "hard_variant_count": 2,
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
