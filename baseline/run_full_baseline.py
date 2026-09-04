from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def find_new_run(before: set[Path]) -> Path:
    candidates = [
        path
        for path in RESULTS_ROOT.glob("tool_use_pilot_*")
        if path.is_dir() and path not in before
    ]
    if not candidates:
        raise RuntimeError("The baseline sweep did not create a new tool_use_pilot_* directory")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    RESULTS_ROOT.mkdir(exist_ok=True)
    existing_runs = set(RESULTS_ROOT.glob("tool_use_pilot_*"))

    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([sys.executable, "run_baseline_sweep.py"])

    run_dir = find_new_run(existing_runs)
    export_dir = run_dir / "exports"
    plots_dir = run_dir / "plots"
    report_path = run_dir / "coordination_report.json"

    run(
        [
            sys.executable,
            "export_trajectory_data.py",
            "--base-dir",
            str(run_dir),
            "--output-dir",
            str(export_dir),
        ]
    )
    run(
        [
            sys.executable,
            "coordination_analysis.py",
            "--base-dir",
            str(run_dir),
            "--output-path",
            str(report_path),
        ]
    )
    run(
        [
            sys.executable,
            "results_visualization.py",
            "--base-dir",
            str(run_dir),
            "--output-dir",
            str(plots_dir),
            "--report-path",
            str(report_path),
        ]
    )

    manifest_path = export_dir / "export_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("issues"):
        raise RuntimeError(f"Trajectory export reported issues: {manifest['issues']}")

    print("\nBaseline run complete")
    print(f"Result directory: {run_dir}")
    print(f"Episodes exported: {manifest.get('episode_count')}")
    print(f"Events exported: {manifest.get('event_count')}")
    print(f"Aggregate summary: {run_dir / 'aggregate_summary.json'}")
    print(f"Coordination report: {report_path}")
    print(f"Dashboards: {plots_dir}")


if __name__ == "__main__":
    main()
