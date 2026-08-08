import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

from settings import get_optional, get_required


load_dotenv()


def normalize_azure_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}"


def build_round1_prompt(mode: str, guess_range: tuple[int, int]) -> str:
    game_desc = "sum guessing game" if mode == "sum" else "number guessing game"
    target_desc = "sum to the mystery number" if mode == "sum" else "average to the mystery number"
    return f"""You are playing a {game_desc}. Your goal is to help your group {target_desc}.
            Your guess range is {guess_range[0]} to {guess_range[1]}.

            Game History:
                This is the first round and you have not made any guesses yet.
            Result: There is no feedback on the group guess yet

            Based on this information, what should your first guess be?

            Respond with only an integer between {guess_range[0]} and {guess_range[1]}."""


def build_round_prompt(mode: str, guess_range: tuple[int, int], round_num: int, history: list[dict]) -> str:
    if round_num == 1:
        return build_round1_prompt(mode, guess_range)

    game_desc = "sum guessing game" if mode == "sum" else "number guessing game"
    target_desc = "sum to the mystery number" if mode == "sum" else "average to the mystery number"

    lines = [
        f"You are playing a {game_desc}. Your goal is to help your group {target_desc}.",
        f"Your guess range is {guess_range[0]} to {guess_range[1]}.",
        "",
        "Game History:",
    ]

    for item in history:
        lines.append(f"Round {item['round']}: your guess used by game = {item['guess']}")
        lines.append(f"Round {item['round']}: your raw response text = {item['raw_content']!r}")
        lines.append(f"Round {item['round']}: result feedback = {item['feedback']}")
        lines.append("")

    lines.extend([
        "Based on this feedback, what should your next guess be?",
        "",
        f"Respond with only an integer between {guess_range[0]} and {guess_range[1]}.",
    ])
    return "\n".join(lines)


def parse_guess(content: str, guess_range: tuple[int, int]) -> int | None:
    numbers = re.findall(r"\d+", content or "")
    if not numbers:
        return None
    value = int(numbers[-1])
    return max(guess_range[0], min(value, guess_range[1]))


def compute_feedback(mode: str, guess: int, mystery_number: int) -> str:
    # Single-agent probe: group aggregate equals the single guess.
    aggregate = guess
    if aggregate == mystery_number:
        return "CORRECT"
    if aggregate > mystery_number:
        return "too HIGH"
    return "too LOW"


def resolve_provider_and_model(model: str) -> tuple[str, str]:
    if model.startswith("openai/"):
        return "openai", model.split("/", 1)[1]
    if model.startswith("azureai/"):
        return "azureai", model.split("/", 1)[1]
    if "/" in model and not model.startswith("gpt-"):
        return "openrouter", model
    return "openai", model


def build_client(provider: str, timeout_seconds: float):
    if provider == "azureai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")
        endpoint = normalize_azure_endpoint(endpoint)
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=get_optional("llm", "azure_openai_api_version", "2025-01-01-preview"),
            timeout=timeout_seconds,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set")
        return AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    raise RuntimeError(f"Unsupported provider for this debug script: {provider}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent raw response probe")
    parser.add_argument("--model", default=get_required("single_run_experiment", "model"))
    parser.add_argument("--mode", default=get_required("single_run_experiment", "mode"), choices=["sum", "mean"])
    parser.add_argument("--temperature", type=float, default=get_required("single_run_experiment", "temperature"))
    parser.add_argument("--min", dest="min_guess", type=int, default=0)
    parser.add_argument("--max", dest="max_guess", type=int, default=50)
    parser.add_argument("--output", default="results/single_agent_probe")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--mystery-number", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")

    provider, resolved_model = resolve_provider_and_model(args.model)
    guess_range = (args.min_guess, args.max_guess)
    mystery_number = args.mystery_number
    if mystery_number is None:
        mystery_number = random.randint(args.min_guess, args.max_guess)
    client = build_client(provider=provider, timeout_seconds=get_optional("llm", "default_timeout_seconds", 45.0))

    os.makedirs(args.output, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output, f"probe_run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    history: list[dict] = []
    summary = {
        "model": args.model,
        "resolved_model": resolved_model,
        "mode": args.mode,
        "temperature": args.temperature,
        "guess_range": [args.min_guess, args.max_guess],
        "rounds": args.rounds,
        "mystery_number": mystery_number,
        "max_completion_tokens": args.max_completion_tokens,
        "results": [],
    }

    for round_num in range(1, args.rounds + 1):
        prompt = build_round_prompt(args.mode, guess_range, round_num, history)
        request = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
        }
        if args.max_completion_tokens is not None:
            request["max_completion_tokens"] = args.max_completion_tokens

        response = await client.chat.completions.create(**request)
        response_data = response.model_dump()
        out_file = os.path.join(run_dir, f"round_{round_num:02d}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2)

        content = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""

        parsed_guess = parse_guess(content, guess_range)
        if parsed_guess is None:
            parsed_guess = (args.min_guess + args.max_guess) // 2
            parse_status = "no_number_found_midpoint_used"
        else:
            parse_status = "ok"

        feedback = compute_feedback(args.mode, parsed_guess, mystery_number)
        history.append(
            {
                "round": round_num,
                "guess": parsed_guess,
                "raw_content": content,
                "feedback": feedback,
            }
        )

        summary["results"].append(
            {
                "round": round_num,
                "response_file": os.path.basename(out_file),
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "raw_content": content,
                "guess_used": parsed_guess,
                "parse_status": parse_status,
                "feedback": feedback,
            }
        )

        print(
            f"Round {round_num}: finish_reason={response.choices[0].finish_reason if response.choices else None}, "
            f"content={content!r}, guess_used={parsed_guess}, feedback={feedback}"
        )

    summary_file = os.path.join(run_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved probe run to: {run_dir}")
    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
