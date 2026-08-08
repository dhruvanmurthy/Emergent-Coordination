from __future__ import annotations

import time

from settings import get_required, require_keys
from tool_use_environment import persist_episode_run, run_demo_episode


def main() -> None:
    section = "single_run_experiment"
    require_keys(section, ["environment", "num_agents", "seed", "step_budget", "results_root"])
    cfg = {
        "environment": get_required(section, "environment"),
        "num_agents": get_required(section, "num_agents"),
        "seed": get_required(section, "seed"),
        "step_budget": get_required(section, "step_budget"),
        "results_root": get_required(section, "results_root"),
    }

    if cfg["environment"] != "synthetic_bugfix":
        raise ValueError(f"Unsupported single-run environment: {cfg['environment']}")

    start_time = time.time()
    environment, events = run_demo_episode(
        seed=cfg["seed"],
        num_agents=cfg["num_agents"],
        step_budget=cfg["step_budget"],
    )
    run_dir = persist_episode_run(
        output_root=cfg["results_root"],
        environment=environment,
        events=events,
    )
    elapsed = time.time() - start_time

    print("Running controlled synthetic bugfix episode from settings.py...")
    print(f"Fixture: {environment.variant.template_id}/{environment.variant.variant_id}")
    print(f"Outcome: {environment.outcome}")
    print(f"Events written to: {run_dir}")
    print(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()