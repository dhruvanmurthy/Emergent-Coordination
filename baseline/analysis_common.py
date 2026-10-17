"""Read and validate saved final-thesis episodes from a ZIP or extracted tree.

No third-party dependencies. This module never invokes an LLM or changes inputs.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

RUN = 'thesis_azureai_gpt-5-mini_20260913_000237'
POLICIES = ('llm_single_agent', 'llm_independent_multi_agent', 'llm_prompted_coordination')
TOOL_NAMES = ('retrieve_file', 'search_symbol', 'apply_patch', 'run_tests', 'finalize_ticket')


@dataclass
class Episode:
    summary: dict[str, Any]
    fixture: dict[str, Any]
    events: list[dict[str, Any]]
    artifact_path: str

    @property
    def condition(self) -> str:
        return self.summary['baseline_name']

    @property
    def fixture_id(self) -> str:
        return f"{self.summary['template_id']}/{self.summary['variant_id']}"


class ArtifactReader:
    """Abstracts reading from a ZIP, a repository tree, or a run directory."""

    def __init__(self, source: str | Path):
        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(f'Input not found: {self.source}')
        self.zf = ZipFile(self.source) if self.source.is_file() and self.source.suffix.lower() == '.zip' else None
        if self.zf is not None:
            names = self.zf.namelist()
            prefixes = sorted({n.split('episodes/', 1)[0] for n in names
                               if RUN in n and '/episodes/' in n and n.endswith('/summary.json')})
            if len(prefixes) != 1:
                raise ValueError(f'Expected one final run in ZIP; found {len(prefixes)}: {prefixes}')
            self.run_prefix = prefixes[0]
            self.episode_dirs = sorted({n.rsplit('/', 1)[0] for n in names
                                       if n.startswith(self.run_prefix + 'episodes/') and n.endswith('/summary.json')})
        else:
            if (self.source / 'episodes').is_dir():
                self.run_root = self.source
            elif (self.source / 'baseline' / 'results' / RUN).is_dir():
                self.run_root = self.source / 'baseline' / 'results' / RUN
            else:
                raise ValueError('Provide THESIS.zip, the extracted repository, or the final run directory')
            self.episode_dirs = [str(p) for p in sorted((self.run_root / 'episodes').iterdir())
                                 if p.is_dir() and (p / 'summary.json').is_file()]

    def _read(self, path: str) -> str:
        if self.zf is not None:
            return self.zf.read(path).decode('utf-8')
        return Path(path).read_text(encoding='utf-8')

    def read_json(self, path: str) -> dict[str, Any]:
        return json.loads(self._read(path))

    def run_json(self, name: str) -> dict[str, Any]:
        p = self.run_prefix + name if self.zf is not None else str(self.run_root / name)
        return self.read_json(p)

    def load(self) -> list[Episode]:
        manifest = self.run_json('manifest.json')
        manifest_by_id = {x['manifest_id']: x for x in manifest['items']}
        if len(manifest_by_id) != len(manifest['items']):
            raise ValueError('Duplicate manifest identifiers')
        episodes = []
        seen_ids = set()
        for directory in self.episode_dirs:
            summary = self.read_json(directory + '/summary.json')
            fixture = self.read_json(directory + '/task_fixture.json')
            raw = [json.loads(x) for x in self._read(directory + '/episode.jsonl').splitlines() if x.strip()]
            if not raw or raw[0].get('record_type') != 'episode_header':
                raise ValueError('Missing episode header: ' + directory)
            events = raw[1:]
            mid = summary['manifest_id']
            if mid in seen_ids or mid not in manifest_by_id:
                raise ValueError('Unexpected or duplicate summary ID: ' + mid)
            seen_ids.add(mid)
            m = manifest_by_id[mid]
            for field, mf in [('baseline_name', 'baseline'), ('seed', 'seed'), ('template_id', 'template_id'),
                              ('variant_id', 'variant_id'), ('repetition', 'repetition')]:
                if summary[field] != m[mf]:
                    raise ValueError(f'Manifest mismatch {mid}: {field}')
            for field in ('template_id', 'variant_id'):
                if fixture[field] != summary[field] or raw[0][field] != summary[field]:
                    raise ValueError(f'Fixture/header mismatch {mid}: {field}')
            if len(events) != summary['event_count'] or len(events) != summary['calls_to_completion']:
                raise ValueError('Event count mismatch: ' + mid)
            if [e.get('sequence_id') for e in events] != list(range(1, len(events) + 1)):
                raise ValueError('Unexpected event sequence: ' + mid)
            if any(e.get('record_type') != 'tool_event' for e in events):
                raise ValueError('Unexpected JSONL record type: ' + mid)
            if any(not isinstance(e.get('agent_id'), int) or
                   e['agent_id'] < 0 or e['agent_id'] >= summary['replay_spec']['num_agents'] for e in events):
                raise ValueError('Out-of-range actor ID: ' + mid)
            tool_counts = Counter(e['tool_name'] for e in events)
            if tool_counts != Counter(summary['tool_counts']):
                raise ValueError('Tool counts disagree with summary: ' + mid)
            if summary['outcome'] not in ('success', 'partial', 'failure'):
                raise ValueError('Unknown outcome: ' + mid)
            episodes.append(Episode(summary, fixture, events, directory))
        if seen_ids != set(manifest_by_id):
            raise ValueError(f'Manifest coverage failure: {len(seen_ids)} / {len(manifest_by_id)}')
        progress = self.run_json('progress.json')
        aggregate = self.run_json('aggregate_summary.json')
        if len(episodes) != progress['completed_count'] or len(episodes) != aggregate['completed_episodes']:
            raise ValueError('Episode count disagrees with progress/aggregate')
        for policy in POLICIES:
            policy_eps = [e for e in episodes if e.condition == policy]
            saved = aggregate['baseline_aggregates'][policy]
            if len(policy_eps) != saved['episodes']:
                raise ValueError('Policy episode count disagrees with aggregate')
            for result in ('success', 'partial', 'failure'):
                observed_rate = sum(e.summary['outcome'] == result for e in policy_eps) / len(policy_eps)
                if abs(observed_rate - saved[result + '_rate']) > 1e-12:
                    raise ValueError(f'{policy} {result} rate disagrees with aggregate')
        return sorted(episodes, key=lambda e: (e.condition, e.fixture_id, e.summary['repetition']))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], fields: list[str], headings: list[str] | None = None) -> str:
    headings = headings or fields
    def clean(v: Any) -> str:
        return str(v).replace('|', r'\|').replace('\n', ' ')
    data = ['| ' + ' | '.join(map(clean, headings)) + ' |',
            '| ' + ' | '.join('---' for _ in fields) + ' |']
    data.extend('| ' + ' | '.join(clean(row.get(f, '')) for f in fields) + ' |' for row in rows)
    return '\n'.join(data)
