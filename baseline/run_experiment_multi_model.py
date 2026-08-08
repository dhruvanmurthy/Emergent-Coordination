from collections import defaultdict
from datetime import datetime
import json
import os
import time

from settings import get_required, require_keys
from tool_use_baselines import run_policy_episode, summarize_baseline_episode, VALID_BASELINES
from tool_use_environment import persist_episode_run


def create_episode_batches(baselines, episode_seeds, batch_size=20):
    all_configs = []
    for baseline_name in baselines:
        for seed in episode_seeds:
            all_configs.append((baseline_name, seed))

    batches = []
    for index in range(0, len(all_configs), batch_size):
        batches.append(all_configs[index:index + batch_size])
    return batches


def save_progress(batch_folder, batch_num, batch_results, total_batches, start_time):
    progress_file = os.path.join(batch_folder, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as handle:
            progress = json.load(handle)
    else:
        progress = {
            "total_batches": total_batches,
            "start_time": start_time,
            "batches_completed": [],
        }

    outcome_counts = defaultdict(int)
    for result in batch_results:
        outcome_counts[result["outcome"]] += 1

    progress["batches_completed"].append(
        {
            "batch_num": batch_num,
            "episode_count": len(batch_results),
            "completed_at": datetime.now().isoformat(),
            "outcome_counts": dict(outcome_counts),
            "results": batch_results,
        }
    )

    with open(progress_file, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)

    return progress


def save_aggregate_summary(batch_folder, progress):
    grouped = defaultdict(list)
    for batch in progress["batches_completed"]:
        for result in batch["results"]:
            grouped[result["baseline_name"]].append(result)

    aggregate = {}
    for baseline_name, results in grouped.items():
        total = len(results)
        aggregate[baseline_name] = {
            "episodes": total,
            "success_rate": sum(1 for result in results if result["success"]) / total if total else 0.0,
            "partial_rate": sum(1 for result in results if result["partial"]) / total if total else 0.0,
            "failure_rate": sum(1 for result in results if result["failure"]) / total if total else 0.0,
            "mean_invalid_call_rate": sum(result["invalid_call_rate"] for result in results) / total if total else 0.0,
            "mean_calls_to_completion": sum(result["calls_to_completion"] for result in results) / total if total else 0.0,
            "mean_patch_acceptance_ratio": sum(result["patch_acceptance_ratio"] for result in results) / total if total else 0.0,
        }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "baseline_aggregates": aggregate,
    }
    with open(os.path.join(batch_folder, "aggregate_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_runner_settings():
    section = "run_experiment_multi_model"
    require_keys(
        section,
        [
            "environment",
            "models",
            "baselines",
            "num_agents",
            "step_budget",
            "episode_seeds",
            "batch_size",
            "results_prefix",
        ],
    )

    models = get_required(section, "models")
    baselines = get_required(section, "baselines")
    if not isinstance(models, list) or not models:
        raise ValueError("run_experiment_multi_model.models must be a non-empty list")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("run_experiment_multi_model.baselines must be a non-empty list")
    unknown = sorted(set(baselines) - VALID_BASELINES)
    if unknown:
        raise ValueError(f"Unknown baselines in settings: {unknown}")

    return {
        "environment": get_required(section, "environment"),
        "models": models,
        "baselines": baselines,
        "num_agents": get_required(section, "num_agents"),
        "step_budget": get_required(section, "step_budget"),
        "episode_seeds": get_required(section, "episode_seeds"),
        "batch_size": get_required(section, "batch_size"),
        "results_prefix": get_required(section, "results_prefix"),
    }


def run_single_episode(baseline_name, seed, batch_folder, model_name, num_agents, step_budget):
    print(f"    Running baseline={baseline_name} seed={seed}")
    environment, events = run_policy_episode(
        baseline_name=baseline_name,
        seed=seed,
        step_budget=step_budget,
        num_agents=num_agents,
    )
    run_dir = persist_episode_run(
        output_root=os.path.join(batch_folder, "episodes"),
        environment=environment,
        events=events,
    )
    summary = summarize_baseline_episode(
        baseline_name=baseline_name,
        environment=environment,
        events=events,
        model_name=model_name,
    )
    summary.update(
        {
            "status": "success",
            "seed": seed,
            "run_dir": run_dir,
        }
    )
    return summary


def run_batch(batch_configs, batch_num, total_batches, batch_folder, model_name, num_agents, step_budget, start_time):
    batch_results = []
    batch_start = time.time()
    print(f"\nBatch {batch_num}/{total_batches} with {len(batch_configs)} episodes")
    for baseline_name, seed in batch_configs:
        batch_results.append(
            run_single_episode(baseline_name, seed, batch_folder, model_name, num_agents, step_budget)
        )

    progress = save_progress(batch_folder, batch_num, batch_results, total_batches, start_time)
    print(f"  Completed in {time.time() - batch_start:.2f}s")
    return progress


def run_for_model(model_name, cfg):
    if cfg["environment"] != "synthetic_bugfix":
        raise ValueError(f"Unsupported batch environment: {cfg['environment']}")

    model_safe_name = model_name.replace("/", "_").replace(":", "_")
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = os.path.join("results", f"{cfg['results_prefix']}_{model_safe_name}_{batch_timestamp}")
    os.makedirs(batch_folder, exist_ok=True)

    batches = create_episode_batches(cfg["baselines"], cfg["episode_seeds"], cfg["batch_size"])
    total_batches = len(batches)
    start_time = time.time()

    experiment_config = {
        "environment": cfg["environment"],
        "model_name": model_name,
        "baselines": cfg["baselines"],
        "num_agents": cfg["num_agents"],
        "step_budget": cfg["step_budget"],
        "episode_seeds": cfg["episode_seeds"],
        "batch_size": cfg["batch_size"],
        "total_episodes": len(cfg["baselines"]) * len(cfg["episode_seeds"]),
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(batch_folder, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(experiment_config, handle, indent=2)

    print(f"Running tool-use baseline sweep for model metadata {model_name}")
    print(f"Results folder: {batch_folder}")

    for batch_num, batch_configs in enumerate(batches, start=1):
        run_batch(
            batch_configs,
            batch_num,
            total_batches,
            batch_folder,
            model_name,
            cfg["num_agents"],
            cfg["step_budget"],
            start_time,
        )

    progress_path = os.path.join(batch_folder, "progress.json")
    with open(progress_path, "r", encoding="utf-8") as handle:
        progress = json.load(handle)
    aggregate = save_aggregate_summary(batch_folder, progress)

    print(f"Completed {len(cfg['baselines']) * len(cfg['episode_seeds'])} episodes")
    print(f"Aggregate summary written to: {os.path.join(batch_folder, 'aggregate_summary.json')}")
    return aggregate


def main():
    cfg = load_runner_settings()
    print("Loaded controlled-environment sweep settings from settings.py")
    for model_name in cfg["models"]:
        run_for_model(model_name, cfg)


if __name__ == "__main__":
    total_start_time = time.time()
    try:
        main()
    finally:
        total_elapsed = time.time() - total_start_time
        print(f"Total program execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")