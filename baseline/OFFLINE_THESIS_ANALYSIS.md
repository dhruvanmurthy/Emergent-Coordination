# Offline thesis experiment analysis

This analysis uses the saved final thesis run at
`results/thesis_azureai_gpt-5-mini_20260913_000237`. It performs:

1. Agent workload analysis: per-agent tool counts, within-episode call imbalance, and descriptive tool-type distribution distance when both agents acted.
2. Fixture-level outcome analysis: outcomes by task family, fixture, and policy/fixture, plus a diagnosis of every partial episode.

These are retrospective analyses of saved observations, not newly executed experiments. The scripts use Python 3.9+ standard-library modules. They do not make network requests, call a model/API, or modify any code.

## Run

From the `baseline` directory:

```bash
python run_all.py "results/thesis_azureai_gpt-5-mini_20260913_000237" --out "results/thesis_offline_analysis"
```

The source run is read-only. The output directory is created if needed and its
analysis files are regenerated on each run.

Individual analyses can also be run from `baseline`:

```bash
python analyze_agent_workload.py "results/thesis_azureai_gpt-5-mini_20260913_000237" --out "results/thesis_offline_analysis"
python analyze_fixture_outcomes.py "results/thesis_azureai_gpt-5-mini_20260913_000237" --out "results/thesis_offline_analysis"
```

## Outputs

| File | Content |
|---|---|
| `RESULTS.md` | Descriptive results, tables, definitions, and caveats |
| `agent_workload_summary.csv` | Policy-level workload metrics |
| `agent_tool_distribution.csv` | Tool counts by agent and two-agent policy |
| `agent_workload_per_episode.csv` | Per-episode workload records for both two-agent policies |
| `fixture_template_summary.csv` | Outcomes aggregated by task family |
| `fixture_variant_summary.csv` | Outcomes for each fixture variant |
| `fixture_by_condition.csv` | Outcomes by policy and fixture |
| `condition_outcome_uncertainty.csv` | Success rates with 95% Wilson intervals |
| `paired_outcome_contrasts.csv` | Paired success-rate differences with fixture-cluster bootstrap intervals |
| `partial_episode_diagnostics.csv` | The five partial episodes and their unresolved defects |

The reader checks manifest coverage and identities, event counts and ordering, tool counts against summaries, fixture/header identities, completion counts, and outcome rates against the saved aggregates. The combined runner also checks the expected dataset totals before writing the report.

## Metric definitions and limitations

**Within-episode workload imbalance** is
`abs(agent_0_calls - agent_1_calls) / (agent_0_calls + agent_1_calls)`. It ranges from 0 (equal participation) to 1 (one active agent). The reported condition mean is an unweighted mean over the episodes for that policy; it is not the imbalance of pooled condition totals.

**Tool-mix total variation distance (TVD)** is half the sum of absolute differences between the agents' normalized tool-type frequencies. It is calculated only when both agents acted and is descriptive, not a validated measure of specialization, teamwork, emergence, or causal coordination.

**Outcome intervals.** Marginal success-rate intervals are two-sided 95% Wilson score intervals. Paired contrasts are success-rate differences in percentage points; their percentile intervals resample the 16 fixture variants as clusters, preserving the four repetitions and three policy assignments per variant (10,000 iterations, fixed seed). The marginal intervals do not model within-fixture dependence. These summaries are descriptive, not significance tests.

**Partial-episode diagnostics** flag an episode when its last event is `finalize_ticket` while required defects remain. This describes the saved trace; it does not establish the model's motivation.

The run has 16 fixture variants, eight task families, four repetitions per condition/variant, and near-ceiling success. The environment measures outcomes on synthetic fixture state, not real code-repair ability. Small per-fixture cells and policy-specific agent-assignment rules mean the workload and outcome differences do not prove that a policy caused better coordination or emergent specialization. In particular, the independent policy alternates preferred agent IDs while prompted coordination allows model-selected IDs.

## Verified source data

The combined run validates 192 completed episodes, 1,181 tool events, 187 successes, five partial outcomes, and no failures against the saved run's manifest, progress file, episode summaries, and aggregate summary. All five partial episodes are in billing fixtures and ended with finalization while a required defect remained and call budget was unused.
