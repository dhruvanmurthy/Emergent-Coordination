# Method and Class Context

This file documents the active controlled tool-use pipeline.

## tool_use_environment.py

1. `build_bugfix_task_fixtures()`
- Builds the deterministic template and variant catalog used in the pilot.

2. `ToolUseBugfixEnvironment.from_seed(...)`
- Chooses a reproducible fixture from the seeded fixture set or loads an explicit template and variant.

3. `ToolUseBugfixEnvironment.list_valid_actions()`
- Enumerates currently valid schema-constrained actions for baseline policies.

4. `ToolUseBugfixEnvironment.step(agent_id, tool_name, arguments)`
- Executes one validated tool call, updates progress, and emits a canonical `ToolEvent`.

5. `ToolUseBugfixEnvironment.summarize_episode(events)`
- Produces episode-level outcome and reliability summaries.

6. `persist_episode_run(...)`
- Writes `episode.jsonl`, `summary.json`, and `task_fixture.json`. Sweep callers provide a baseline-and-seed label to avoid overwriting repeated fixture runs and persist the enriched baseline summary.

## tool_use_baselines.py

1. `build_policy_adapter(...)`
- Selects the requested baseline policy implementation.

2. `run_policy_episode(...)`
- Runs one controlled episode for a specific baseline and seed.

3. `summarize_baseline_episode(...)`
- Adds policy-facing metrics such as action counts, patch acceptance ratio, and calls to completion.

## coordination_metrics.py

1. `mutual_information(...)`
- Computes categorical MI in bits using either the plug-in or Miller-Madow estimator.

2. `smoothing_sensitivity(...)`
- Reports MI under multiple smoothing settings.

3. `bootstrap_confidence_interval(...)`
- Produces bootstrap confidence intervals for MI.

4. `permutation_null_distribution(...)`
- Estimates a null distribution by shuffling outcomes and computes a permutation p-value.

Current interpretation constraint:
- Full action traces are currently used as categorical MI inputs. The pilot showed high trace cardinality and constant outcomes for scripted baselines, so MI conclusions are deferred until coarsened features and sanity checks are added.

## coordination_analysis.py

1. `collect_episode_records(base_dir)`
- Loads canonical episode artifacts and derives per-episode coordination features.

2. `encode_agent_trace(events, agent_id)`
- Converts an agent's abstraction sequence into a categorical trace token.

3. `summarize_coordination(base_dir)`
- Aggregates baseline-level MI, reliability, efficiency, bootstrap, and permutation-null summaries.
- Single-agent episodes currently fill the second MI source from the first agent and should therefore be treated as a separate reference condition, not as an equivalent two-agent comparison.

## export_trajectory_data.py

1. `collect_trajectory_rows(base_dir)`
- Converts canonical episode artifacts into row-oriented event and summary tables.

2. `write_optional_parquet(rows, output_path)`
- Writes Parquet when the runtime provides a compatible pandas/parquet stack.

3. `export_trajectory_tables(base_dir, output_dir)`
- Emits CSV tables, optional Parquet tables, and an export manifest.

## results_visualization.py

1. `create_dashboard(report, output_dir)`
- Renders PDF dashboards for MI, invalid-call rate, completion efficiency, and recovery behavior.

2. `main()`
- Rebuilds the coordination report and writes visualization artifacts.

## debug_single_agent.py

1. `load_latest_episode(base_dir)`
- Resolves the newest controlled episode run under a results directory.

2. `describe_episode(run_dir)`
- Loads `episode.jsonl` and `summary.json` and prints a compact per-agent trace summary.

3. `main()`
- CLI entrypoint for inspecting a single controlled episode.

## experiment.py

1. `main()`
- Runs a single deterministic controlled bugfix episode from `settings.py`.

## run_baseline_sweep.py

1. `main()`
- Runs the baseline sweep over deterministic seeds and writes per-episode artifacts plus aggregate summaries.
