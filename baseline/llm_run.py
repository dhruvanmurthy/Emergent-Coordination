import asyncio
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AsyncAzureOpenAI, AsyncOpenAI, BadRequestError, RateLimitError
from settings import get_optional, get_required


load_dotenv()


OPENROUTER_BASE_URL = get_required("llm", "openrouter_base_url")
AZURE_OPENAI_API_VERSION = get_optional("llm", "azure_openai_api_version", "2025-01-01-preview")
DEFAULT_TIMEOUT_SECONDS = get_optional("llm", "default_timeout_seconds", 45.0)
DEFAULT_MAX_RETRIES = get_optional("llm", "default_max_retries", 5)
DEFAULT_BASE_DELAY_SECONDS = get_optional("llm", "default_base_delay_seconds", 1.0)
DEFAULT_FALLBACK_GUESS = get_optional("llm", "default_fallback_guess", 25)


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
    """OpenAI-compatible lightweight response for graceful degradation."""

    def __init__(self, model: str, content: str, reason: str):
        self.id = f"fallback-{int(time.time() * 1000)}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = [_Choice(index=0, message=_Message(role="assistant", content=content), finish_reason="stop")]
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
                    "index": c.index,
                    "message": asdict(c.message),
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": asdict(self.usage),
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
        }


def _resolve_provider_and_model(model: str) -> tuple[str, str]:
    """
    Resolve provider + model mapping.

    Supported patterns:
    - openai/gpt-4o-mini -> provider=openai, model=gpt-4o-mini
    - gpt-4o-mini -> provider=openai, model=gpt-4o-mini
    - meta-llama/llama-3.3-70b-instruct -> provider=openrouter, model=meta-llama/llama-3.3-70b-instruct
    - google/gemini-2.5-flash -> provider=openrouter, model=google/gemini-2.5-flash
    """
    if model.startswith("openai/"):
        return "openai", model.split("/", 1)[1]

    if model.startswith("azureai/"):
        return "azureai", model.split("/", 1)[1]

    if "/" in model and not model.startswith("gpt-"):
        return "openrouter", model

    return "openai", model


def _build_client(provider: str, timeout_seconds: float) -> AsyncOpenAI:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        return AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment.")
        return AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=timeout_seconds)

    if provider == "azureai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set in the environment.")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set in the environment.")
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=AZURE_OPENAI_API_VERSION,
            timeout=timeout_seconds,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _build_extra_headers_for_openrouter() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
    app_title = os.getenv("OPENROUTER_APP_TITLE")

    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if app_title:
        headers["X-Title"] = app_title

    return headers


def _uses_max_completion_tokens(model_name: str) -> bool:
    """GPT-5 family requires max_completion_tokens instead of max_tokens."""
    normalized = (model_name or "").lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4")


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, asyncio.TimeoutError)):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code >= 500:
        return True

    return False


async def _call_chat_completion(
    client: AsyncOpenAI,
    provider: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    if _uses_max_completion_tokens(model_name):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    if provider == "openrouter":
        extra_headers = _build_extra_headers_for_openrouter()
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

    return await client.chat.completions.create(**kwargs)


def _make_fallback_response(model: str, reason: str, fallback_guess: int = DEFAULT_FALLBACK_GUESS) -> FallbackResponse:
    # Keep output parse-friendly for downstream numeric extraction.
    return FallbackResponse(model=model, content=str(fallback_guess), reason=reason)


async def chat(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 32,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    soft_fallback: bool = True,
    fallback_guess: int = DEFAULT_FALLBACK_GUESS,
) -> Any:
    """
    Unified async LLM chat function used by experiments.

    Returns:
      - OpenAI SDK response object on success
      - FallbackResponse on final failure when soft_fallback=True

    This shape is intentionally compatible with code that reads:
      response.choices[0].message.content
      response.model_dump()  # optional
      response.is_fallback   # optional
    """
    provider, resolved_model = _resolve_provider_and_model(model)

    try:
        client = _build_client(provider=provider, timeout_seconds=timeout_seconds)
    except Exception as exc:
        if soft_fallback:
            return _make_fallback_response(model=model, reason=f"client_init_error: {exc}", fallback_guess=fallback_guess)
        raise

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await _call_chat_completion(
                client=client,
                provider=provider,
                model_name=resolved_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response

        except BadRequestError as exc:
            # Prompt/model/input issue is usually non-retryable.
            last_error = exc
            break

        except Exception as exc:
            last_error = exc
            if not _is_retryable_exception(exc) or attempt == max_retries:
                break

            # Exponential backoff with jitter.
            sleep_seconds = base_delay_seconds * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, base_delay_seconds)
            await asyncio.sleep(sleep_seconds + jitter)

    if soft_fallback:
        return _make_fallback_response(
            model=model,
            reason=f"request_failed_after_retries: {type(last_error).__name__}: {last_error}",
            fallback_guess=fallback_guess,
        )

    raise RuntimeError(
        f"chat() failed for provider={provider}, model={model}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )
