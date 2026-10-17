#!/usr/bin/env python3
"""Run both saved-data analyses and verify the final report remains untouched."""
import argparse
from pathlib import Path
from collections import Counter
from analysis_common import markdown_table, TOOL_NAMES, POLICIES
from analysis_common import ArtifactReader, RUN
from analyze_agent_workload import analyze as workload
from analyze_fixture_outcomes import analyze as fixtures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', help='Original THESIS.zip or an extracted thesis directory')
    ap.add_argument('--out', default='results', help='Where CSVs and RESULTS.md are written')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = ArtifactReader(args.source)
    eps = archive.load()
    assert len(eps) == 192, len(eps)
    assert sum(len(e.events) for e in eps) == 1181
    assert Counter(e.summary['outcome'] for e in eps) == Counter(success=187, partial=5)
    balance = workload(eps, out)
    by_template, billing, partials, rates, contrasts = fixtures(eps, out)
    actor_tool_rows = []
    for tool in TOOL_NAMES:
        row = {'tool': tool}
        for policy, prefix in [(POLICIES[1], 'independent'), (POLICIES[2], 'prompted')]:
            active = [e for e in eps if e.condition == policy]
            for actor in (0, 1):
                row[f'{prefix}_agent_{actor}'] = sum(
                    1 for episode in active for event in episode.events
                    if event['agent_id'] == actor and event['tool_name'] == tool)
        actor_tool_rows.append(row)
    actor_tool_table = markdown_table(
        actor_tool_rows,
        ['tool', 'independent_agent_0', 'independent_agent_1',
         'prompted_agent_0', 'prompted_agent_1'],
        ['Tool', 'Independent A0', 'Independent A1', 'Prompted A0', 'Prompted A1'])
    assert len(partials) == 5
    assert all(p['finalized_early_with_defect'] and p['unused_calls'] > 0 for p in partials)
    assert all(p['fixture'].startswith('billing/') for p in partials)
    content = f'''# Offline thesis experiment results

Source: `{Path(args.source).name}`, final run `{RUN}`.

The scripts read the 192 completed episode summaries, 1,181 JSONL tool events, 192 task fixtures and run manifest. Results are independently recalculated from episode artifacts and reconciled with `aggregate_summary.json`. No LLM API calls or report edits are made.

## 1. Agent workload analysis

{balance}

**Definitions.** Per-episode imbalance = `abs(agent_0_calls - agent_1_calls) / (agent_0_calls + agent_1_calls)`. Values range from 0 (equal call counts) to 1 (only one agent acts). Mean imbalance is an unweighted mean over all 64 episodes per two-agent policy. Tool-mix TVD (total variation distance) = `0.5 * sum(abs(p0(tool)-p1(tool)))` across tool names; it is calculated only for episodes where both agents acted. Neither metric proves specialization or causal coordination. Agent 0 and agent 1 are policy actor IDs, not distinguishable human participants. The policy assignment rules themselves may produce imbalance differences.

### Actor-level tool types

{actor_tool_table}

These tool counts describe actor assignments, not verified roles or intentions. For example, the prompted policy assigns 109 patch calls to actor 0 and 26 to actor 1, while assigning 36 test calls to actor 0 and 64 to actor 1. Since the independent policy forces alternating actor IDs and the prompted policy allows selecting agent IDs, the conditions are not matched interventions for studying emergent workload differences.

## 2. Fixture-level outcomes

{by_template}

### Condition success rates and paired contrasts

{rates}

The condition-level 95% intervals are two-sided Wilson score intervals for the marginal episode proportions; they do not separately model within-fixture dependence. Paired differences are in success percentage points (A minus B). Their 95% percentile intervals resample the 16 fixture variants as clusters (10,000 iterations), preserving all four repetitions and the three policy assignments within each sampled variant. These are descriptive uncertainty summaries, not hypothesis tests.

{contrasts}

### Billing outcomes by condition

{billing}

All five partial episodes occur in billing variants and end with `finalize_ticket` while at least one required defect remains. Each uses fewer than 12 allowed tool calls. This identifies the **immediate event-level reason** for the partial results, not the model's underlying decision process or proof that coordination caused the results.

| Condition | Fixture | Repetition | Calls used | Unused budget | Remaining required defects |
| --- | --- | ---: | ---: | ---: | --- |
'''
    for partial in sorted(partials, key=lambda x: (x['condition'], x['fixture'], x['repetition'])):
        content += (f"| {partial['condition']} | {partial['fixture']} | {partial['repetition']} | "
                    f"{partial['calls_used']} | {partial['unused_calls']} | "
                    f"{partial['remaining_required_defects']} |\n")
    content += '''
## Validity and interpretation

- This is **retrospective analysis of saved observations**, not two new randomized LLM experiments.
- The final run includes 16 fixture variants (eight families) and 4 repetitions per condition/variant.
- Agent-workload analysis includes only the two-agent conditions (64 episodes each). The single-agent condition is used as an outcome reference but not a workload-balance comparator.
- The environment supplies a set of allowable actions; submitted-action validity and simulated task success should not be interpreted as real code-repair ability.
- Small per-fixture cell sizes (four repetitions) and near-ceiling success mean no claim of statistical superiority or generalization follows from these descriptive tables.
- Paired contrasts preserve the shared fixture/repetition assignments and resample fixture variants as clusters; no significance tests are added.
- The original thesis report has not been modified.

## Output files

- `agent_workload_summary.csv`: policy-level workload summary.
- `agent_tool_distribution.csv`: per-agent tool action counts.
- `agent_workload_per_episode.csv`: every two-agent episode, for audit/reanalysis.
- `fixture_template_summary.csv`: outcomes aggregated over task family.
- `fixture_variant_summary.csv`: outcomes for each of 16 variants.
- `fixture_by_condition.csv`: condition x fixture outcomes.
- `condition_outcome_uncertainty.csv`: condition success rates with Wilson intervals.
- `paired_outcome_contrasts.csv`: paired success-rate differences with fixture-cluster bootstrap intervals.
- `partial_episode_diagnostics.csv`: exact five partial episodes.
'''
    (out / 'RESULTS.md').write_text(content, encoding='utf-8')
    print(f'PASS: verified {len(eps)} completed episodes, '
          f'{sum(len(e.events) for e in eps)} tool events, '
          f'{sum(e.summary["outcome"] == "partial" for e in eps)} partial outcomes')
    print(balance + '\n\n' + by_template + '\n\n' + billing)
    print('\nCondition success uncertainty:\n' + rates)
    print('\nPaired contrasts:\n' + contrasts)
    print(f'\nCreated outputs in: {out}')


if __name__ == '__main__':
    main()
