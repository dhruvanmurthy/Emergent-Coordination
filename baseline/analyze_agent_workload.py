#!/usr/bin/env python3
"""Analyze agent participation and tool-use specialization from saved thesis episodes.

Usage: python analyze_agent_workload.py /path/to/THESIS.zip --out results
"""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
from analysis_common import ArtifactReader, TOOL_NAMES, POLICIES, write_csv, markdown_table


def analyze(episodes, out: Path):
    rows = []
    tool_rows = []
    episode_rows = []
    for condition in POLICIES[1:]:
        selected = [e for e in episodes if e.condition == condition]
        balance = []
        tvds = []
        total_by_actor = Counter()
        total_by_actor_tool = Counter()
        single_active = 0
        both_active = 0
        for ep in selected:
            actor_counts = Counter(e['agent_id'] for e in ep.events)
            actor_tools = Counter((e['agent_id'], e['tool_name']) for e in ep.events)
            n0, n1 = actor_counts[0], actor_counts[1]
            total = n0 + n1
            assert total == len(ep.events) and total > 0
            imbalance = abs(n0 - n1) / total
            if n0 == 0 or n1 == 0:
                single_active += 1
                tvd = None  # An inactive agent has no observed tool-choice distribution.
            else:
                both_active += 1
                tvd = 0.5 * sum(abs(actor_tools[(0,t)] / n0 - actor_tools[(1,t)] / n1) for t in TOOL_NAMES)
                tvds.append(tvd)
            balance.append(imbalance)
            total_by_actor.update(actor_counts)
            total_by_actor_tool.update(actor_tools)
            er = {'condition': condition, 'fixture': ep.fixture_id,
                  'manifest_id': ep.summary['manifest_id'], 'repetition': ep.summary['repetition'],
                  'seed': ep.summary['seed'], 'outcome': ep.summary['outcome'],
                  'total_calls': total, 'agent_0_calls': n0, 'agent_1_calls': n1,
                  'imbalance': round(imbalance, 8), 'single_active_agent': int(min(n0,n1) == 0),
                  'tool_mix_tvd_when_both_active': '' if tvd is None else round(tvd, 8)}
            for a in (0,1):
                for t in TOOL_NAMES:
                    er[f'agent_{a}_{t}'] = actor_tools[(a,t)]
            episode_rows.append(er)
        n = len(selected)
        total_calls = total_by_actor[0] + total_by_actor[1]
        rows.append({'condition': condition, 'episodes': n, 'total_tool_calls': total_calls,
                     'agent_0_calls': total_by_actor[0], 'agent_1_calls': total_by_actor[1],
                     'agent_0_share_pct': round(100 * total_by_actor[0] / total_calls, 2),
                     'agent_1_share_pct': round(100 * total_by_actor[1] / total_calls, 2),
                     'mean_imbalance_pct': round(100 * mean(balance), 2),
                     'median_imbalance_pct': round(100 * median(balance), 2),
                     'single_active_episodes': single_active,
                     'both_active_episodes': both_active,
                     'mean_tool_mix_tvd_both_active': round(mean(tvds), 4) if tvds else ''})
        for t in TOOL_NAMES:
            tool_rows.append({'condition': condition, 'tool': t, 'agent_0': total_by_actor_tool[(0,t)],
                              'agent_1': total_by_actor_tool[(1,t)],
                              'total': sum(total_by_actor_tool[(a,t)] for a in (0,1))})
    write_csv(out / 'agent_workload_summary.csv', rows, list(rows[0]))
    write_csv(out / 'agent_tool_distribution.csv', tool_rows, list(tool_rows[0]))
    write_csv(out / 'agent_workload_per_episode.csv', episode_rows, list(episode_rows[0]))
    display = []
    for r in rows:
        display.append({'condition': r['condition'].replace('llm_', '').replace('_multi_agent','').replace('_', ' '),
                        'agent_0_calls': r['agent_0_calls'], 'agent_1_calls': r['agent_1_calls'],
                        'mean_imbalance_pct': f"{r['mean_imbalance_pct']:.2f}%",
                        'single_active_episodes': f"{r['single_active_episodes']}/{r['episodes']}",
                        'mean_tool_mix_tvd_both_active': f"{r['mean_tool_mix_tvd_both_active']:.3f}"})
    return markdown_table(display, ['condition','agent_0_calls','agent_1_calls','mean_imbalance_pct',
                                    'single_active_episodes','mean_tool_mix_tvd_both_active'],
                          ['Policy', 'Agent 0 calls', 'Agent 1 calls', 'Mean imbalance', 'One active agent',
                           'Mean tool-mix TVD*'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='THESIS.zip, extracted thesis repository, or final thesis run directory')
    parser.add_argument('--out', default='results', help='Output directory (default: results)')
    args = parser.parse_args()
    episodes = ArtifactReader(args.source).load()
    table = analyze(episodes, Path(args.out))
    print(table)
    print('\n* TVD compares each active agent\'s distribution across tool types; '
          'it is descriptive, not a validated specialization or emergence score.')


if __name__ == '__main__':
    main()
