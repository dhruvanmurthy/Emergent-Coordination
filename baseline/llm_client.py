from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from settings import get_optional


load_dotenv()


DEFAULT_LLM_MODEL = os.getenv(
    "LLM_MODEL",
    get_optional("llm", "default_model", "azureai/gpt-5-mini"),
)
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION",
    get_optional("llm", "azure_openai_api_version", "2025-01-01-preview"),
)
DEFAULT_TIMEOUT_SECONDS = float(
    os.getenv("LLM_TIMEOUT_SECONDS", get_optional("llm", "default_timeout_seconds", 45.0))
)
DEFAULT_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", get_optional("llm", "default_max_retries", 5))
)
DEFAULT_BASE_DELAY_SECONDS = float(
    os.getenv("LLM_BASE_DELAY_SECONDS", get_optional("llm", "default_base_delay_seconds", 1.0))
)
DEFAULT_MAX_COMPLETION_TOKENS = int(
    os.getenv("LLM_MAX_COMPLETION_TOKENS", get_optional("llm", "max_completion_tokens", 256))
)
DEFAULT_ACTION_PARSE_ATTEMPTS = int(
    os.getenv("LLM_ACTION_PARSE_ATTEMPTS", get_optional("llm", "action_parse_attempts", 2))
)


@dataclass
class _Message:
    role: str
    content: str


@dataclass
class _Choice:
    index: int
    message: _Message
    finish_reason: Optional[str] = None


@dataclass
class _FallbackUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FallbackResponse:
    """OpenAI-compatible response used when a request cannot be completed."""

    def __init__(self, model: str, content: str, reason: str):
        self.id = f"fallback-{int(time.time() * 1000)}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = [
            _Choice(
                index=0,
                message=_Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ]
        self.usage = _FallbackUsage()
        self.is_fallback = True
        self.fallback_reason = reason

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": asdict(choice.message),
                    "finish_reason": choice.finish_reason,
                }
                for choice in self.choices
            ],
            "usage": asdict(self.usage),
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
        }


def _resolve_provider_and_model(model: str) -> Tuple[str, str]:
    if model.startswith("openai/"):
        return "openai", model.split("/", 1)[1]
    if model.startswith("azureai/"):
        return "azureai", model.split("/", 1)[1]
    if "/" in model and not model.startswith("gpt-"):
        return "openrouter", model
    return "openai", model


def _normalize_azure_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
            break
    return normalized


def _load_openai_sdk() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        from openai import APIConnectionError, APITimeoutError, AzureOpenAI, OpenAI, RateLimitError
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for LLM-backed baselines.") from exc
    return OpenAI, AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError


def _build_client(provider: str, timeout_seconds: float) -> Any:
    OpenAI, AzureOpenAI, _, _, _ = _load_openai_sdk()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        return OpenAI(api_key=api_key, timeout=timeout_seconds)

    if provider == "azureai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in the environment.")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set in the environment.")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=_normalize_azure_endpoint(azure_endpoint),
            api_version=AZURE_OPENAI_API_VERSION,
            timeout=timeout_seconds,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _uses_max_completion_tokens(model_name: str) -> bool:
    normalized = (model_name or "").lower()
    return (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )


def _is_retryable_exception(exc: Exception) -> bool:
    try:
        _, _, APIConnectionError, APITimeoutError, RateLimitError = _load_openai_sdk()
        retryable_types = (APIConnectionError, APITimeoutError, RateLimitError, TimeoutError)
    except RuntimeError:
        retryable_types = (TimeoutError,)

    if isinstance(exc, retryable_types):
        return True

    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500


def _call_chat_completion(
    client: Any,
    model_name: str,
    prompt: str,
    max_completion_tokens: int,
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if _uses_max_completion_tokens(model_name):
        kwargs["max_completion_tokens"] = max_completion_tokens
    else:
        kwargs["max_tokens"] = max_completion_tokens
    return client.chat.completions.create(**kwargs)


def chat_json(
    model: str,
    prompt: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    soft_fallback: bool = True,
) -> Any:
    """Call a JSON-mode chat completion with synchronous retry handling."""
    provider, resolved_model = _resolve_provider_and_model(model)

    try:
        client = _build_client(provider=provider, timeout_seconds=timeout_seconds)
    except Exception as exc:
        if soft_fallback:
            return FallbackResponse(
                model=model,
                content='{"tool_name": null, "arguments": {}}',
                reason=f"client_init_error: {exc}",
            )
        raise

    last_error: Optional[Exception] = None
    attempt_count = max(int(max_retries), 1)
    for attempt in range(1, attempt_count + 1):
        try:
            return _call_chat_completion(
                client=client,
                model_name=resolved_model,
                prompt=prompt,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as exc:
            last_error = exc
            if not _is_retryable_exception(exc) or attempt == attempt_count:
                break
            sleep_seconds = base_delay_seconds * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, base_delay_seconds)
            time.sleep(sleep_seconds + jitter)

    if soft_fallback:
        return FallbackResponse(
            model=model,
            content='{"tool_name": null, "arguments": {}}',
            reason=f"request_failed_after_retries: {type(last_error).__name__}: {last_error}",
        )

    raise RuntimeError(
        f"chat_json() failed for provider={provider}, model={model}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )