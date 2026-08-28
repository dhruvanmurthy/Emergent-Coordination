# Emergent Coordination Baseline

This repository now runs a controlled tool-use coordination study instead of the earlier scalar guessing game. The active system is a deterministic synthetic bugfix environment with strict tool schemas, explicit invalid-call reasons, ternary outcomes, and count-based mutual information analysis.

## Active Scope

- Primary task family: synthetic bugfix workflow.
- Primary analysis setting: 2-agent coordination.
- Included pilot baselines: `single_agent`, `random_policy`, `independent_multi_agent`, `prompted_coordination`.
- Allowed tools: `retrieve_file`, `search_symbol`, `run_tests`, `apply_patch`, `finalize_ticket`.
- Episode outcomes: `success`, `partial`, `failure`.

## Quick Start

```bash
pip install -r requirements.txt
python experiment.py
python run_baseline_sweep.py
python export_trajectory_data.py --base-dir results --output-dir results/exports
python results_visualization.py --base-dir results --output-dir results/plots
```

## Core Pipeline

1. `experiment.py` runs a deterministic demo episode from `settings.py`.
2. `run_baseline_sweep.py` sweeps baseline policies over fixed seeds.
3. `tool_use_environment.py` validates each tool call and emits canonical event logs.
4. `coordination_analysis.py` aggregates episode traces into MI-ready categorical records.
5. `results_visualization.py` produces coordination, reliability, and efficiency dashboards.

## Outputs

Per-episode artifacts:

```text
results/tool_use_run_TEMPLATE_VARIANT_LABEL_TIMESTAMP/
├── episode.jsonl
├── summary.json
└── task_fixture.json
```

For sweep runs, `LABEL` contains the baseline and seed, which keeps repeated runs for the same fixture separate.

Batch artifacts:

```text
results/tool_use_pilot_MODEL_TIMESTAMP/
├── experiment_config.json
├── progress.json
├── aggregate_summary.json
└── episodes/
```

Analysis exports:

```text
results/exports/
├── trajectory_events.csv
├── trajectory_summaries.csv
├── trajectory_events.parquet
├── trajectory_summaries.parquet
└── export_manifest.json
```

## Notes

- The scalar-game workflow is retired from the active code path.
- Sweep episode directories include the baseline and seed so repeated fixture selections are not overwritten.
- The current 80-episode run is a pipeline-validation pilot, not a final research result. Scripted baselines have constant success outcomes, and random-policy traces are high-cardinality.
- Pending execution-time and research-validity checks are tracked in `validation.md`.


