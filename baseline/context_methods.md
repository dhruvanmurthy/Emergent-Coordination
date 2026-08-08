# Method and Class Context

This file documents each significant function/method in the codebase with intent, inputs, outputs, side effects, and error handling.

## llm_run.py
- Purpose: unified async LLM API wrapper with retry and fallback.

### Dataclasses and response compatibility types

1. _Message
- Purpose: lightweight message container for fallback responses.
- Fields: role, content.

2. _Choice
- Purpose: single completion choice wrapper in fallback response.
- Fields: index, message, finish_reason.

3. _FallbackUsage
- Purpose: placeholder token usage for synthetic responses.
- Fields: prompt_tokens, completion_tokens, total_tokens.

4. FallbackResponse.__init__(model, content, reason)
- Purpose: create OpenAI-like response object when real calls fail.
- Inputs:
  - model: requested model string
  - content: assistant text payload (numeric guess as string)
  - reason: human-readable fallback reason
- Output: instance with id/object/created/model/choices/usage plus is_fallback metadata.
- Error handling: no explicit handling; constructor is simple field assembly.

5. FallbackResponse.model_dump()
- Purpose: emit JSON-serializable dict matching OpenAI-ish structure.
- Output: dict with standard completion fields plus fallback metadata.

### Provider and client resolution

6. _resolve_provider_and_model(model)
- Purpose: map user model string to provider + provider-specific model name.
- Inputs: model string.
- Output: tuple(provider, resolved_model).
- Rules:
  - openai/xyz -> (openai, xyz)
  - non-gpt prefixed strings containing slash -> openrouter
  - otherwise -> openai

7. _build_client(provider, timeout_seconds)
- Purpose: instantiate AsyncOpenAI client for selected provider.
- Inputs: provider, timeout_seconds.
- Output: AsyncOpenAI instance.
- Error handling:
  - raises ValueError when required API key env var is missing
  - raises ValueError on unsupported provider

8. _build_extra_headers_for_openrouter()
- Purpose: optional attribution headers for OpenRouter.
- Output: dict with HTTP-Referer and X-Title when environment variables exist.

### Retry and request layer

9. _is_retryable_exception(exc)
- Purpose: classify whether request errors should be retried.
- Retryable:
  - RateLimitError
  - APITimeoutError
  - APIConnectionError
  - asyncio.TimeoutError
  - exceptions with integer status_code >= 500
- Output: bool.

10. _call_chat_completion(client, provider, model_name, prompt, temperature, max_tokens)
- Purpose: make one chat completion API call.
- Inputs: request parameters.
- Output: provider SDK response object.
- Side effects: network I/O.

11. _make_fallback_response(model, reason, fallback_guess=25)
- Purpose: create parse-friendly fallback completion.
- Output: FallbackResponse whose content is fallback_guess as string.

12. chat(model, prompt, temperature=0.7, max_tokens=32, timeout_seconds=45.0, max_retries=5, base_delay_seconds=1.0, soft_fallback=True, fallback_guess=25)
- Purpose: unified async chat entry point used across experiments.
- Inputs:
  - model and prompt required
  - generation/retry/fallback options optional
- Output:
  - real SDK response on success
  - FallbackResponse if soft_fallback=True and failures persist
- Error handling:
  - client init errors convert to fallback when soft_fallback=True
  - BadRequestError treated as non-retryable and exits retry loop
  - retryable errors use exponential backoff + jitter
  - when soft_fallback=False, raises RuntimeError with last error details

## experiment.py
- Purpose: core game logic for non-persona experiments.

### Types

1. ParsingError
- Purpose: signals failed numeric parsing from model output.

2. Round dataclass
- Purpose: immutable-ish per-round state record.
- Key fields:
  - guesses, average, rounded_average, feedback
  - prompts_used
  - api_response_ids
  - parsing_failures
  - fallback_responses

### Agent class

3. Agent.__init__(agent_id, model, temperature)
- Purpose: initialize one participant.
- State initialized:
  - guess_history list
  - last_successful_guess for fallback reuse

4. Agent.make_guess(round_num, game_history, guess_range, mode)
- Purpose: generate one integer guess for a round.
- Inputs:
  - round number
  - prior Round list
  - allowed range tuple
  - mode
- Output: tuple(guess, prompt_used, response_obj, parsing_failed_flag).
- Workflow:
  - builds first-round or strategic prompt
  - captures prompt via prompt_capture.capture_prompt
  - calls llm_run.chat
  - parses numeric response
- Error handling:
  - on ParsingError, reuses last successful guess if available
  - if no prior guess, falls back to midpoint

5. Agent._build_strategic_prompt(game_history, guess_range, mode)
- Purpose: include own previous guesses and feedback for iterative strategy.
- Output: full prompt string.

6. Agent._extract_number(response, guess_range)
- Purpose: wrapper to robust extraction logic.

7. Agent._extract_number_robust(response, guess_range)
- Purpose: multi-strategy parse pipeline.
- Strategy order:
  - _extract_last_number
  - _extract_first_number
  - _extract_any_number (clamped)
  - midpoint fallback generator

8. Agent._extract_last_number(response, guess_range)
- Purpose: parse last integer token and validate range.
- Error handling: raises ParsingError if none/invalid.

9. Agent._extract_first_number(response, guess_range)
- Purpose: parse first integer token and validate range.
- Error handling: raises ParsingError if none/invalid.

10. Agent._extract_any_number(response, guess_range)
- Purpose: parse first integer and clamp into range.
- Error handling: raises ParsingError if no integer found.

11. Agent._generate_fallback(guess_range)
- Purpose: deterministic midpoint fallback.

12. Agent._get_response_content(response)
- Purpose: normalize content extraction across response shapes.
- Supports:
  - response.message.content
  - response.choices[0].message.content
  - str(response) fallback

### GameMaster class

13. GameMaster.__init__(mode='mean', mystery_range=None, temperature=0.7, max_rounds=20, num_agents=None, batch_folder=None, run_id=1)
- Purpose: initialize game environment and output directories.
- Inputs: experiment mode/configuration.
- Side effects:
  - chooses mystery number
  - creates results folder
  - initializes prompt capture
  - writes config.json

14. GameMaster.add_agent(model)
- Purpose: append Agent configured with game temperature.
- Output: Agent instance.

15. GameMaster._save_config()
- Purpose: persist run config metadata.
- Output file: config.json.

16. GameMaster._save_round(round_data)
- Purpose: persist per-round JSON snapshot.
- Output file: round_XX.json.

17. GameMaster._update_log(round_data)
- Purpose: append concise textual line to game_log.txt.
- Includes parsing and API fallback annotations when present.

18. GameMaster._log_parsing_failure(round_num, agent_id, error_msg)
- Purpose: append dedicated parsing failure line to log.
- Note: currently not used in active round flow.

19. GameMaster._save_parsing_failure(round_num, agent_id, error_msg)
- Purpose: write structured parsing_failure.json.
- Note: currently not used in active round flow.

20. GameMaster.play_round(round_num)
- Purpose: execute one complete round concurrently across all agents.
- Output: Round object.
- Steps:
  - gather agent guesses concurrently
  - track parsing_failures and fallback_responses
  - compute sum/mean result and feedback
  - append game history and persist files
- Error handling:
  - no outer try in round execution; agent and chat-level fallbacks absorb many failures
  - API response save has per-agent try/except warning path

21. GameMaster.play_game()
- Purpose: iterate rounds until solved or max rounds reached.
- Output: list of Round objects.
- Side effects:
  - prints progress
  - writes final summary + CSV
  - saves captured prompts

22. GameMaster._save_final_summary()
- Purpose: write summary.json and trigger CSV export.

23. GameMaster._save_csv_summary()
- Purpose: write round-by-round per-agent CSV.
- Output file pattern: game_data - Target: X .csv

### Module-level runner

24. run_async_test(num_agents=5, model='gpt-4o-mini', temperature=0.7, mode='mean', run_id=1)
- Purpose: convenience async single-run harness.
- Output: results directory path.

25. __main__ block
- Purpose: default local run for one sum-mode test.

## prompt_capture.py
- Purpose: selective prompt capture into llm_prompts.json.

1. PromptCapture.__init__(results_dir)
- Purpose: initialize selective prompt capture store and output path.

2. PromptCapture.capture_prompt(round_num, agent_id, prompt)
- Purpose: store prompt only for configured capture rounds.
- Default rounds: 1,2,5,10,15,20.

3. PromptCapture.save_prompts()
- Purpose: write llm_prompts.json.

4. PromptCapture.print_captured_prompts()
- Purpose: intentionally no-op (console printing removed).

5. init_prompt_capture(results_dir)
- Purpose: initialize module-global prompt capture singleton.

6. capture_prompt(round_num, agent_id, prompt)
- Purpose: proxy to singleton if initialized.

7. save_and_display_prompts()
- Purpose: currently save only.

## persona_wrapper.py
- Purpose: persona loading, assignment, and prompt augmentation.

1. PersonaWrapper.__init__(persona_file='personas_gpt41.txt')
- Purpose: load persona definitions and initialize mapping.

2. PersonaWrapper._load_personas(file_path)
- Inputs: path to text file with one persona per line.
- Output: list of non-empty stripped lines.
- Side effects: prints count.
- Error handling: relies on Python file I/O exceptions.

3. PersonaWrapper.assign_personas(num_agents)
- Purpose: shuffle personas and assign one per agent id.
- Output: dict agent_id -> persona text.
- Caveat: assumes enough persona lines for num_agents.

4. PersonaWrapper.enhance_prompt(agent_id, original_prompt)
- Purpose: prepend persona text when assignment exists.
- Output: modified or original prompt.

## persona_experiment.py
- Purpose: batch runner variant using persona-enhanced prompts and parsing strategy.

### PersonaAgent

1. PersonaAgent.__init__(agent_id, model, temperature, persona_wrapper)
- Purpose: Agent subclass with persona support.

2. PersonaAgent.make_guess(round_num, game_history, guess_range, mode)
- Purpose: use persona-augmented prompts and reasoning-style response format.
- Output: tuple compatible with Agent.make_guess.
- Differences from base Agent:
  - prompts ask for reasoning and FINAL GUESS marker
  - max_tokens is higher (200)
  - exceptions are re-raised instead of local fallback reuse

3. PersonaAgent._build_strategic_prompt(...)
- Purpose: reasoning-focused strategic prompt builder.

4. PersonaAgent._extract_number(response, guess_range)
- Purpose: delegate to FINAL GUESS parser.

5. PersonaAgent._extract_final_guess(response, guess_range)
- Purpose: regex parse of FINAL GUESS: N, with range check.
- Fallback: calls base robust parser if marker absent/invalid.

### PersonaGameMaster

6. PersonaGameMaster.__init__(..., persona_wrapper=None)
- Purpose: GameMaster subclass carrying persona mapping.

7. PersonaGameMaster.add_agent(model)
- Purpose: create PersonaAgent instances.

8. PersonaGameMaster._save_config()
- Purpose: config writer that includes persona text for each agent.

### Batch orchestration functions

9. run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds)
- Purpose: run one persona configuration.
- Output: status dict.
- Error handling:
  - ParsingError -> parsing_failed status
  - generic Exception -> failed status

10. create_fallback_result(config_id, agents, temp, run_id, error_msg)
- Purpose: build synthetic failed result payload.

11. create_config_batches(agents_list, temp_list, runs_per_config, batch_size=20)
- Purpose: enumerate config grid and chunk into batches.

12. save_progress(batch_folder, batch_num, batch_results, total_batches)
- Purpose: append batch summary into progress.json.

13. load_progress(batch_folder)
- Purpose: load progress file if present.

14. run_batch(batch_configs, batch_num, total_batches, batch_folder, model, client_type, mode, max_rounds, max_concurrent=10)
- Purpose: run one batch with bounded concurrency.
- Error handling:
  - asyncio.gather(return_exceptions=True)
  - converts exceptions into failed records

15. main()
- Purpose: CLI entrypoint for persona experiment sweeps.
- Responsibilities:
  - parse model arg
  - set parameter grid
  - create results folder and config
  - run batches, support resume semantics
  - print final aggregate summary

16. __main__ wrapper
- Purpose: robust run with KeyboardInterrupt and fatal exception handling.

## run_experiment_multi_model.py
- Purpose: large-scale non-persona sweep orchestrator.

1. run_single_config(...)
- Purpose: non-persona single config runner.
- Error handling:
  - ParsingError -> parsing_failed
  - generic exception -> failed_with_fallback using create_fallback_result

2. create_fallback_result(...)
- Purpose: normalize catastrophic failures in result stream.

3. create_config_batches(...)
- Purpose: grid expansion + batching.

4. save_progress(...)
- Purpose: append batch outcomes to progress.json.

5. save_failure_summary(batch_folder, progress)
- Purpose: produce parsing_failures_analysis.json with rates and groupings.

6. load_progress(batch_folder)
- Purpose: resume support.

7. run_batch(...)
- Purpose: bounded-concurrency batch executor.

8. main()
- Purpose: CLI entrypoint for large non-persona sweeps.
- Behavior:
  - expects exactly one model name argument
  - defines parameter lists
  - creates massive_experiment_... folder
  - runs all batches and prints final summary

9. __main__ wrapper
- Purpose: top-level run timing and fatal exception reporting.

## extract_game_data_to_csv.py
- Purpose: converts selected run logs to success/failure CSV snapshots.

1. extract_round_data(content)
- Purpose: parse Round N: guesses=[...] lines from game_log text.
- Input: full log content string.
- Output: sorted list of rows [round, guess1, guess2, ...].
- Error handling: no explicit try/except in parser itself.

2. save_to_csv(rounds_data, filename, exp_name, run_name, success_type)
- Purpose: write extracted rows with inferred agent headers.
- Output: CSV file.

3. extract_game_data(base_dir, experiment_names)
- Purpose: scan run folders for first success and first failure per experiment, then export two CSV files.
- Error handling:
  - missing logs are skipped
  - file read errors are caught and printed
  - missing round data logged and skipped

4. Module-level execution
- Defines experiment_names and base_directory then immediately runs extract_game_data.

## results_visualization.py
- Purpose: aggregate runs and generate analysis plots as PDFs.

1. parse_game_log(file_path)
- Purpose: classify run as converged / not converged / parsing failure and extract convergence rounds.
- Output: tuple(converged_bool, rounds_or_none, parsing_failure_bool).
- Error handling: broad except returns parsing_failure=True.

2. collect_results(base_path)
- Purpose: aggregate counts by (agents, temperature) config folder.
- Output: dict keyed by (agents,temp) with counts and round lists.

3. create_plots(results, base_path)
- Purpose: generate six PDF visualizations in base_path/plots.
- Inputs: aggregated results and path.
- Output: plot files.
- Error handling:
  - early returns on empty/invalid data
  - conditional plotting guards for empty matrices

4. Module-level execution
- Uses hardcoded base_path and runs collect_results/create_plots when path exists.

## results_compresser.py
- Purpose: compress results folders (zip or tar.xz).

1. compress_folder_zip(folder_path, output_name='results_compressed.zip')
- Purpose: recursively zip folder with max compressionlevel 9.
- Output: zip file and printed size stats.

2. compress_folder_tar_xz(folder_path, output_name='results_compressed.tar.xz')
- Purpose: create tar.xz archive at preset 9.
- Output: tar.xz file and printed size stats.

3. Module-level execution
- currently calls compress_folder_tar_xz('./results') directly.

## launch_experiments.sh

- Purpose: start one tmux session per configured model and run run_experiment_multi_model.py in each.
- Behavior:
  - sanitizes session names
  - prints attach/kill helper commands

## README.md

- Purpose: project overview, usage, experiment scales, and output format notes.
- Not executable but describes expected behavior and structure.

# File and Artifact Inventory Context

## Results Folder: Per-Run Artifacts

### Folder: results/experiment_run_20260627_111324

1. config.json
- Run configuration snapshot.
- Keys: mystery_number, mystery_range, guess_range, mode, temperature, max_rounds, timestamp, agents.

2. game_log.txt
- Human-readable round log with guesses, aggregate value, feedback, and API failure annotations.

3. raw_api_api_r05_a0.json
4. raw_api_api_r05_a2.json
5. raw_api_api_r05_a4.json
6. raw_api_api_r06_a0.json
7. raw_api_api_r06_a1.json
8. raw_api_api_r06_a2.json
9. raw_api_api_r06_a3.json
10. raw_api_api_r06_a4.json
11. raw_api_api_r07_a0.json
12. raw_api_api_r07_a1.json
13. raw_api_api_r07_a2.json
14. raw_api_api_r07_a3.json
15. raw_api_api_r07_a4.json
- Purpose: saved API responses for selected rounds/agents.
- Observed structure in sample file:
  - id, object, created, model, choices, usage
  - plus fallback metadata is_fallback and fallback_reason for fallback responses.
- Naming pattern:
  - raw_api_api_rRR_aA.json where RR is round and A is agent id.

16. round_01.json
17. round_02.json
18. round_03.json
19. round_04.json
20. round_05.json
21. round_06.json
22. round_07.json
- Purpose: full per-round structured state.
- Common keys:
  - round_num, guesses, average, rounded_average, feedback, mystery_number, mode
  - prompts_used
  - api_response_ids (nullable)
  - parsing_failures (nullable)
  - fallback_responses (nullable)

## sampled_data Folder

These are compact per-round CSV snapshots extracted from larger experiments.

1. sampled_data/google_gemini-2.5-flash-preview-05-20_20250703_221915_FAILURE.csv
2. sampled_data/google_gemini-2.5-flash-preview-05-20_20250703_221915_SUCCESS.csv
3. sampled_data/meta-llama_llama-3.3-70b-instruct_20250703_221915_FAILURE.csv
4. sampled_data/meta-llama_llama-3.3-70b-instruct_20250703_221915_SUCCESS.csv
5. sampled_data/openai_gpt-4o-mini_20250702_182448_FAILURE.csv
6. sampled_data/openai_gpt-4o-mini_20250702_182448_SUCCESS.csv

Common CSV schema:
- Header: round, agent_1, agent_2, ...
- Rows: one round per row, integer guesses per agent.

## Data Quality and Behavioral Notes

1. Fallback API responses are explicitly persisted and identifiable via is_fallback=true in raw API JSON.

2. Some runs include no summary.json (example: experiment_run_20260627_111324), indicating interrupted or partial completion.

3. Round files and log files are the strongest source-of-truth for per-step behavior.

4. Empty artifact file exists:
- results/experiment_run_20260627_150318/game_data - Target

## Suggested Reading Order for New Contributors

1. context.md
2. context_methods.md
3. experiment.py
4. llm_run.py
5. run_experiment_multi_model.py
6. persona_experiment.py
7. context_artifacts.md for output schemas and file references

## Operational Notes and Examples

### Entry Points in This Repository

The codebase has two useful definitions of entry points:

1. Python main-guard entry points (3)
- experiment.py
- persona_experiment.py
- run_experiment_multi_model.py

2. Directly runnable scripts including top-level execution (7)
- experiment.py
- persona_experiment.py
- run_experiment_multi_model.py
- extract_game_data_to_csv.py
- results_visualization.py
- results_compresser.py
- launch_experiments.sh

### Running Experiments with OpenRouter Free Models

The OpenRouter path is already implemented in llm_run.py via:
- _resolve_provider_and_model
- _build_client
- _build_extra_headers_for_openrouter

#### Required Environment Variables

Example .env entries:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_HTTP_REFERER=https://github.com/your-org/your-repo
OPENROUTER_APP_TITLE=Emergent-Coordination
```

Notes:
- OPENROUTER_API_KEY is required for provider=openrouter calls.
- OPENROUTER_HTTP_REFERER and OPENROUTER_APP_TITLE are optional but recommended.

#### Model Routing Behavior in _resolve_provider_and_model

Examples with expected routing:

1. OpenAI direct
- Input: openai/gpt-4o-mini
- Output: provider=openai, model=gpt-4o-mini

2. OpenAI implicit
- Input: gpt-4o-mini
- Output: provider=openai, model=gpt-4o-mini

3. OpenRouter model (recommended format)
- Input: meta-llama/llama-3.3-70b-instruct:free
- Output: provider=openrouter, model=meta-llama/llama-3.3-70b-instruct:free

4. OpenRouter model (another example)
- Input: google/gemma-2-9b-it:free
- Output: provider=openrouter, model=google/gemma-2-9b-it:free

#### CLI Examples

Run non-persona sweep with an OpenRouter free model:

```bash
python run_experiment_multi_model.py "meta-llama/llama-3.3-70b-instruct:free"
```

Run persona experiment with an OpenRouter free model:

```bash
python persona_experiment.py "google/gemma-2-9b-it:free"
```

#### Concurrency Tuning for Free-Tier Stability

Free models often rate-limit aggressively. Recommended starting settings:

1. run_experiment_multi_model.py
- max_concurrent = 2 or 3
- runs_per_config = small trial first (for example 3 to 5)

2. persona_experiment.py
- max_concurrent = 2 or 3
- runs_per_config = small trial first

3. Keep retry/fallback enabled
- llm_run.chat already includes retry + exponential backoff + soft fallback.

#### Optional Resolver Enhancement

If you want explicit support for openrouter/<model> input style, add this case:

```python
if model.startswith("openrouter/"):
  return "openrouter", model.split("/", 1)[1]
```

Example:
- Input: openrouter/meta-llama/llama-3.3-70b-instruct:free
- Output: provider=openrouter, model=meta-llama/llama-3.3-70b-instruct:free
