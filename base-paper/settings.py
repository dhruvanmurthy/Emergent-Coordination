from typing import Any, Dict, Iterable


# Centralized runtime configuration for the experiment scripts.
# These comments preserve the intent of the earlier inline defaults.
SETTINGS: Dict[str, Any] = {
    # Shared LLM client settings used by the chat wrapper and experiment runners.
    "llm": {
        # OpenRouter endpoint for OpenAI-compatible model routing.
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        # Default timeout for chat completion requests.
        "default_timeout_seconds": 45.0,
        # Retry attempts for transient API failures.
        "default_max_retries": 5,
        # Initial backoff delay before retrying a failed request.
        "default_base_delay_seconds": 5.0,
        # Fallback numeric guess used when the model call fails.
        "default_fallback_guess": 25,
    },
    # Large multi-model sweep configuration used by run_experiment_multi_model.py.
    "run_experiment_multi_model": {
        # Models to evaluate in the batch experiment.
        "models": ["openrouter/free"],
        # Client type expected by the runner.
        "client_type": "openai",
        # Aggregation strategy for the game result.
        "mode": "sum",
        # Maximum number of rounds per game.
        "max_rounds": 20,
        # Agent counts to sweep across.
        "agents_list": [3],
        # Temperatures to sweep across.
        "temp_list": [round(0.1 * i, 1) for i in range(0, 6)],
        # Repetitions per (agents, temperature) configuration.
        "runs_per_config": 5,
        # Number of configurations processed in one batch.
        "batch_size": 10,
        # Maximum number of concurrent runs for the batch job.
        "max_concurrent": 3,
        # Optional batch number to resume from; None starts fresh.
        "resume_from_batch": None,
        # Prefix used for generated result directories and files.
        "results_prefix": "massive_experiment",
    },
    # Persona-based experiment settings used by persona_experiment.py.
    "persona_experiment": {
        # Models to evaluate in the persona experiment.
        "models": ["openrouter/free"],
        # Client type expected by the runner.
        "client_type": "openai",
        # Aggregation strategy for the game.
        "mode": "sum",
        # Maximum rounds for each simulation.
        "max_rounds": 20,
        # Agent counts to sweep across.
        "agents_list": [3],
        # Temperature used for the persona runs.
        "temp_list": [1.0],
        # Repetitions per configuration.
        "runs_per_config": 5,
        # Configurations processed in one batch.
        "batch_size": 10,
        # Maximum number of concurrent runs.
        "max_concurrent": 3,
        # Optional batch number to resume from.
        "resume_from_batch": None,
        # Prefix used for generated experiment outputs.
        "results_prefix": "persona+reasoning_experiment",
        # Distinguishes this run as a persona experiment.
        "experiment_type": "persona_experiment",
        # File containing the persona prompts.
        "persona_file": "personas_gpt41.txt",
        # Max token budget used for persona-enhanced agent replies.
        "persona_agent_max_tokens": 200,
    },
    # Minimal single-run example configuration.
    "single_run_experiment": {
        # Number of agents in the sample run.
        "num_agents": 5,
        # Model used for the demo game.
        "model": "openrouter/free",
        # Temperature for the single run.
        "temperature": 1.9,
        # Aggregation strategy for the sample game.
        "mode": "sum",
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
