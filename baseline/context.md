# Repository Context

## Purpose
This repository runs controlled synthetic bugfix coordination experiments and analyzes whether joint multi-agent tool-use behavior predicts task outcomes.

Core research question: estimate $I(A_1, A_2; Y)$ for categorical agent traces and ternary episode outcomes in a deterministic environment.

## High-Level Architecture

1. Controlled environment and baselines
- experiment.py
- tool_use_environment.py
- tool_use_baselines.py
- run_baseline_sweep.py

2. Analysis and reporting
- coordination_metrics.py
- coordination_analysis.py
- export_trajectory_data.py
- results_visualization.py

3. Configuration and docs
- settings.py
- validation.md
- README.md
- context_methods.md

## End-to-End Execution Flow

1. `experiment.py` runs one deterministic demo episode.

2. `run_baseline_sweep.py` executes baseline sweeps across fixed seeds.

3. `ToolUseBugfixEnvironment` validates tool calls and writes canonical artifacts.
- `episode.jsonl`
- `summary.json`
- `task_fixture.json`

4. Offline scripts consume those canonical artifacts directly.
- `export_trajectory_data.py` exports row-oriented event and summary tables
- `coordination_analysis.py` computes MI-ready baseline summaries
- `results_visualization.py` renders dashboards from the coordination report

## Core Concepts

1. Progress states
- `not_started`
- `in_progress`
- `completed_success`
- `completed_partial`
- `completed_failure`

2. Abstraction categories
- `retrieve`
- `verify`
- `update`
- `finalize`
- `error_or_noop`

3. Invalid-call taxonomy
- `schema_invalid`
- `reference_invalid`
- `state_invalid`
- `semantic_noop`
- `budget_invalid`

## Practical Notes

1. The scalar guessing workflow is retired from the mainline implementation.

2. The pipeline has been validated on an 80-episode sweep: all episode directories were retained, 558 events were exported, and the coordination report and dashboards were generated.

3. The current MI output is diagnostic rather than publishable: scripted baselines have constant success outcomes, while random-policy full traces have nearly one unique joint trace per episode. Coarsened trace features and MI sanity checks are still required.

4. Single-agent episodes are a reference condition and should not be interpreted as a two-agent mutual-information comparison.

5. Remaining execution-time and research-validity checks are tracked in `validation.md`.
