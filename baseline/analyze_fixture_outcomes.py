#!/usr/bin/env python3
"""Summarize outcomes by task template/variant and diagnose incomplete episodes.

Usage: python analyze_fixture_outcomes.py /path/to/THESIS.zip --out results
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
from statistics import mean
from analysis_common import ArtifactReader, POLICIES, write_csv, markdown_table


def wilson_interval(successes, trials, confidence_level=0.95):
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError('trials must be positive')
    z = 1.959963984540054 if confidence_level == 0.95 else None
    if z is None:
        raise ValueError('Only a 95% confidence interval is supported')
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * ((proportion * (1 - proportion) / trials
                       + z * z / (4 * trials * trials)) ** 0.5) / denominator
    return center - half_width, center + half_width


def _percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def outcome_uncertainty(episodes, iterations=10000, seed=20261017):
    """Summarize marginal rates and paired, fixture-cluster bootstrap contrasts."""
    by_condition = defaultdict(list)
    by_fixture = defaultdict(dict)
    for episode in episodes:
        summary = episode.summary
        condition = episode.condition
        fixture = episode.fixture_id
        repetition = summary['repetition']
        by_condition[condition].append(episode)
        cell = by_fixture[fixture].setdefault(repetition, {})
        if condition in cell:
            raise ValueError(f'Duplicate condition/repetition assignment: {fixture}, {repetition}')
        cell[condition] = {
            'success': summary['outcome'] == 'success',
            'seed': summary['seed'],
        }

    fixture_ids = sorted(by_fixture)
    if not fixture_ids:
        raise ValueError('No episodes supplied')
    repetitions = set()
    for fixture, fixture_cells in by_fixture.items():
        if len(fixture_cells) != 4:
            raise ValueError(f'Expected four repetitions for {fixture}; found {len(fixture_cells)}')
        repetitions.update(fixture_cells)
        for repetition, assignments in fixture_cells.items():
            if set(assignments) != set(POLICIES):
                raise ValueError(f'Incomplete paired assignment: {fixture}, repetition {repetition}')
            seeds = {assignments[condition]['seed'] for condition in POLICIES}
            if len(seeds) != 1:
                raise ValueError(f'Mismatched seeds: {fixture}, repetition {repetition}')
    if len(repetitions) != 4:
        raise ValueError(f'Expected four shared repetitions; found {sorted(repetitions)}')

    rate_rows = []
    for condition in POLICIES:
        selected = by_condition[condition]
        successes = sum(episode.summary['outcome'] == 'success' for episode in selected)
        low, high = wilson_interval(successes, len(selected))
        rate_rows.append({
            'condition': condition,
            'episodes': len(selected),
            'successes': successes,
            'success_rate_pct': round(100 * successes / len(selected), 3),
            'wilson_95_low_pct': round(100 * low, 3),
            'wilson_95_high_pct': round(100 * high, 3),
            'partial': sum(episode.summary['outcome'] == 'partial' for episode in selected),
            'failure': sum(episode.summary['outcome'] == 'failure' for episode in selected),
        })

    comparisons = (
        (POLICIES[0], POLICIES[1]),
        (POLICIES[0], POLICIES[2]),
        (POLICIES[2], POLICIES[1]),
    )
    rng = random.Random(seed)
    bootstrap_differences = {pair: [] for pair in comparisons}
    for _ in range(iterations):
        sampled_fixtures = [rng.choice(fixture_ids) for _ in fixture_ids]
        sampled_successes = {condition: 0 for condition in POLICIES}
        sampled_n = 0
        for fixture in sampled_fixtures:
            for assignments in by_fixture[fixture].values():
                sampled_n += 1
                for condition in POLICIES:
                    sampled_successes[condition] += int(assignments[condition]['success'])
        sampled_rates = {condition: sampled_successes[condition] / sampled_n
                         for condition in POLICIES}
        for condition_a, condition_b in comparisons:
            bootstrap_differences[(condition_a, condition_b)].append(
                100 * (sampled_rates[condition_a] - sampled_rates[condition_b]))

    contrast_rows = []
    point_rates = {row['condition']: row['success_rate_pct'] for row in rate_rows}
    for condition_a, condition_b in comparisons:
        values = sorted(bootstrap_differences[(condition_a, condition_b)])
        contrast_rows.append({
            'condition_a': condition_a,
            'condition_b': condition_b,
            'success_rate_difference_pp': round(point_rates[condition_a] - point_rates[condition_b], 3),
            'cluster_bootstrap_95_low_pp': round(_percentile(values, 0.025), 3),
            'cluster_bootstrap_95_high_pp': round(_percentile(values, 0.975), 3),
            'fixture_clusters': len(fixture_ids),
            'bootstrap_iterations': iterations,
            'bootstrap_seed': seed,
        })
    return rate_rows, contrast_rows


def analyze(episodes, out: Path):
    by_template = defaultdict(list)
    by_variant = defaultdict(list)
    by_condition_variant = defaultdict(list)
    partial_rows = []
    for e in episodes:
        s = e.summary
        by_template[s['template_id']].append(e)
        by_variant[e.fixture_id].append(e)
        by_condition_variant[(e.fixture_id, e.condition)].append(e)
        if s['outcome'] == 'partial':
            required = set(e.fixture['required_defects'])
            fixed = set(s['fixed_defects'])
            remaining = sorted(required - fixed)
            last = e.events[-1] if e.events else {}
            partial_rows.append({'condition': e.condition, 'fixture': e.fixture_id,
                                 'repetition': s['repetition'], 'seed': s['seed'],
                                 'manifest_id': s['manifest_id'],
                                 'calls_used': len(e.events), 'step_budget': s['step_budget'],
                                 'unused_calls': s['step_budget'] - len(e.events),
                                 'last_tool': last.get('tool_name', ''),
                                 'remaining_required_defects': '; '.join(remaining),
                                 'remaining_defect_count': len(remaining),
                                 'finalized_early_with_defect': int(last.get('tool_name') == 'finalize_ticket' and bool(remaining))})
    def outcome_row(label, selected, field):
        c = Counter(x.summary['outcome'] for x in selected)
        return {field: label, 'episodes': len(selected), 'success': c['success'],
                'partial': c['partial'], 'failure': c['failure'],
                'success_pct': round(100*c['success']/len(selected), 2),
                'partial_pct': round(100*c['partial']/len(selected), 2),
                'mean_tool_calls': round(mean(e.summary['calls_to_completion'] for e in selected), 3)}
    template_rows = [outcome_row(k,v,'template') for k,v in sorted(by_template.items())]
    variant_rows = [outcome_row(k,v,'fixture') for k,v in sorted(by_variant.items())]
    condition_rows = [dict({'condition': cond}, **outcome_row(fix, vals, 'fixture'))
                      for (fix,cond),vals in sorted(by_condition_variant.items())]
    rate_rows, contrast_rows = outcome_uncertainty(episodes)
    write_csv(out / 'condition_outcome_uncertainty.csv', rate_rows, list(rate_rows[0]))
    write_csv(out / 'paired_outcome_contrasts.csv', contrast_rows, list(contrast_rows[0]))
    write_csv(out / 'fixture_template_summary.csv', template_rows, list(template_rows[0]))
    write_csv(out / 'fixture_variant_summary.csv', variant_rows, list(variant_rows[0]))
    write_csv(out / 'fixture_by_condition.csv', condition_rows, list(condition_rows[0]))
    write_csv(out / 'partial_episode_diagnostics.csv', sorted(partial_rows, key=lambda x:(x['condition'],x['fixture'],x['repetition'])), list(partial_rows[0]))
    overview = []
    for r in template_rows:
        overview.append({'template': r['template'], 'episodes': r['episodes'],
                         'success': r['success'], 'partial': r['partial'], 'failure':r['failure'],
                         'success_pct': f"{r['success_pct']:.2f}%"})
    table = markdown_table(overview, ['template','episodes','success','partial','failure','success_pct'],
                           ['Task family','Episodes','Success','Partial','Failure','Success rate'])
    # Billing by condition for quick focused comparison
    billing = [e for e in episodes if e.summary['template_id']=='billing']
    focused = []
    for cond in POLICIES:
        sel = [e for e in billing if e.condition == cond]
        c = Counter(e.summary['outcome'] for e in sel)
        focused.append({'condition': cond, 'success': c['success'], 'partial': c['partial'],
                        'n': len(sel), 'success_rate': f"{100*c['success']/len(sel):.1f}%"})
    focused_table = markdown_table(focused,['condition','n','success','partial','success_rate'],
                                  ['Condition','Billing episodes','Success','Partial','Success rate'])
    rates_table = markdown_table(
        [{'condition': row['condition'],
          'success': f"{row['successes']}/{row['episodes']}",
          'success_rate': f"{row['success_rate_pct']:.2f}%",
          'wilson_ci': f"[{row['wilson_95_low_pct']:.2f}%, {row['wilson_95_high_pct']:.2f}%]"}
         for row in rate_rows],
        ['condition', 'success', 'success_rate', 'wilson_ci'],
        ['Condition', 'Successes', 'Success rate', '95% Wilson CI'])
    contrasts_table = markdown_table(
        [{'contrast': f"{row['condition_a']} - {row['condition_b']}",
          'difference': f"{row['success_rate_difference_pp']:+.2f} pp",
          'ci': f"[{row['cluster_bootstrap_95_low_pp']:+.2f}, "
                f"{row['cluster_bootstrap_95_high_pp']:+.2f}] pp"}
         for row in contrast_rows],
        ['contrast', 'difference', 'ci'],
        ['Paired contrast (A - B)', 'Success-rate difference', '95% fixture-cluster bootstrap CI'])
    return table, focused_table, partial_rows, rates_table, contrasts_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='THESIS.zip, extracted repository, or final thesis run directory')
    parser.add_argument('--out', default='results', help='Output directory (default: results)')
    args = parser.parse_args()
    episodes = ArtifactReader(args.source).load()
    by_template, focused, partial, rates, contrasts = analyze(episodes, Path(args.out))
    print('By task family:\n'+by_template+'\n\nBilling only:\n'+focused)
    print('\nCondition-level success uncertainty:\n'+rates)
    print('\nPaired success-rate contrasts:\n'+contrasts)
    print(f'\nPartial episodes: {len(partial)}; premature finalizations with unfixed requirements: '
          f'{sum(p["finalized_early_with_defect"] for p in partial)}')


if __name__ == '__main__':
    main()
