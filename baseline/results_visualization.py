from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from coordination_analysis import summarize_coordination


def _ordered_baselines(report: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return sorted(report.get("baselines", {}).items(), key=lambda item: item[0])


def create_dashboard(report: Dict[str, Any], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    baseline_items = _ordered_baselines(report)
    if not baseline_items:
        return []

    labels = [name for name, _ in baseline_items]
    mi_items = [
        (name, item)
        for name, item in baseline_items
        if item.get("two_agent_mi", {}).get("status") == "computed"
        and item.get("two_agent_mi", {}).get("mi_bits") is not None
    ]
    mi_labels = [name for name, _ in mi_items]
    mi_bits = [item["two_agent_mi"]["mi_bits"] for _, item in mi_items]
    invalid_rates = [item["mean_invalid_call_rate"] for _, item in baseline_items]
    calls_to_completion = [item["mean_calls_to_completion"] for _, item in baseline_items]
    patch_acceptance = [item["mean_patch_acceptance_ratio"] for _, item in baseline_items]
    recovery_rates = [item["mean_invalid_recovery_rate"] for _, item in baseline_items]
    ci_low = [item["two_agent_mi"]["bootstrap"]["lower_bound"] for _, item in mi_items]
    ci_high = [item["two_agent_mi"]["bootstrap"]["upper_bound"] for _, item in mi_items]
    yerr = np.array([
        [mi - low for mi, low in zip(mi_bits, ci_low)],
        [high - mi for mi, high in zip(mi_bits, ci_high)],
    ])

    x = np.arange(len(labels))
    x_mi = np.arange(len(mi_labels))
    saved_paths: List[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(x_mi, mi_bits, color=["#2a9d8f", "#e9c46a", "#f4a261", "#264653"][: len(mi_labels)])
    if mi_bits:
        axes[0, 0].errorbar(x_mi, mi_bits, yerr=yerr, fmt="none", ecolor="#111111", capsize=6)
    axes[0, 0].set_xticks(x_mi)
    axes[0, 0].set_xticklabels(mi_labels, rotation=20, ha="right")
    axes[0, 0].set_ylabel("MI bits")
    axes[0, 0].set_title("Coordination by Baseline")

    axes[0, 1].bar(x, invalid_rates, color="#b56576")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Invalid call rate")
    axes[0, 1].set_title("Reliability")

    axes[1, 0].bar(x, calls_to_completion, color="#577590")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Calls to completion")
    axes[1, 0].set_title("Efficiency")

    axes[1, 1].scatter(patch_acceptance, recovery_rates, s=120, c=np.linspace(0.2, 0.8, len(labels)), cmap="viridis")
    for index, label in enumerate(labels):
        axes[1, 1].annotate(label, (patch_acceptance[index], recovery_rates[index]), textcoords="offset points", xytext=(6, 4))
    axes[1, 1].set_xlabel("Patch acceptance ratio")
    axes[1, 1].set_ylabel("Invalid recovery rate")
    axes[1, 1].set_title("Recovery vs Patch Quality")

    fig.tight_layout()
    dashboard_path = os.path.join(output_dir, "tool_use_dashboard.pdf")
    fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(dashboard_path)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    width = 0.25
    test_efficiency = [item["mean_test_run_efficiency"] for _, item in mi_items]
    p_values = [item["two_agent_mi"]["permutation_null"].get("p_value", 0.0) for _, item in mi_items]
    ax2.bar(x_mi - width, mi_bits, width=width, label="MI bits", color="#43aa8b")
    ax2.bar(x_mi, test_efficiency, width=width, label="Test efficiency", color="#f9c74f")
    ax2.bar(x_mi + width, p_values, width=width, label="Permutation p-value", color="#f94144")
    ax2.set_xticks(x_mi)
    ax2.set_xticklabels(mi_labels, rotation=20, ha="right")
    ax2.set_title("Coordination Summary")
    ax2.legend()
    fig2.tight_layout()
    summary_path = os.path.join(output_dir, "coordination_summary.pdf")
    fig2.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(summary_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visualization dashboards for controlled tool-use runs.")
    parser.add_argument("--base-dir", default="results", help="Root directory containing controlled tool-use outputs.")
    parser.add_argument("--output-dir", default=os.path.join("results", "plots"), help="Directory to write PDF dashboards.")
    parser.add_argument("--report-path", default=os.path.join("results", "coordination_report.json"), help="Path to persist the coordination summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_coordination(args.base_dir)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    saved_paths = create_dashboard(report, args.output_dir)
    print(f"Saved {len(saved_paths)} dashboards to {args.output_dir}")


if __name__ == "__main__":
    main()