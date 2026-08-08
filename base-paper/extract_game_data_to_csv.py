import os
import re
import csv
import json
import argparse
import glob

def load_json_file(path):
    """Load JSON file and return dict/list or None on error."""
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to read JSON {path}: {e}")
        return None


def extract_round_data_from_log(content):
    """Extract round rows from game log text."""
    rounds_data = []
    pattern = r"Round\s+(\d+):\s+guesses=\[(.*?)\]"

    for match in re.finditer(pattern, content):
        round_num = int(match.group(1))
        guesses_str = match.group(2)
        guesses = [int(x) for x in re.findall(r"-?\d+", guesses_str)]
        row = [round_num] + guesses
        rounds_data.append(row)

    rounds_data.sort(key=lambda x: x[0])
    return rounds_data


def extract_round_metadata_from_log(content):
    """Extract per-round metadata from game log text when JSON round files are absent."""
    metadata = {}
    line_pattern = re.compile(
        r"Round\s+(\d+):\s+guesses=\[(.*?)\]"
        r"(?:,\s*(sum|avg)=(-?\d+(?:\.\d+)?))?"
        r"(?:,\s*rounded=(-?\d+))?"
        r"(?:,\s*(.*))?$"
    )

    for line in content.splitlines():
        match = line_pattern.match(line.strip())
        if not match:
            continue

        round_num = int(match.group(1))
        metric_name = match.group(3)
        metric_value = match.group(4)
        rounded_value = match.group(5)
        feedback = (match.group(6) or "").strip()

        if "parsing_failures=" in feedback:
            feedback = feedback.split(", parsing_failures=", 1)[0].strip()
        if "api_failures=" in feedback:
            feedback = feedback.split(", api_failures=", 1)[0].strip()

        metadata[round_num] = {
            "group_metric_name": metric_name,
            "group_metric_value": float(metric_value) if metric_value is not None else None,
            "rounded_group_value": int(rounded_value) if rounded_value is not None else None,
            "feedback": feedback,
            "parsing_failures_count": None,
            "fallback_responses_count": None,
        }

    return metadata


def extract_round_data_from_json(run_dir):
    """Extract round rows + metadata from round_XX.json files."""
    round_paths = sorted(glob.glob(os.path.join(run_dir, "round_*.json")))
    if not round_paths:
        return [], {}

    rounds_data = []
    round_meta = {}

    for round_path in round_paths:
        payload = load_json_file(round_path)
        if not isinstance(payload, dict):
            continue

        round_num = payload.get("round_num")
        if round_num is None:
            m = re.search(r"round_(\d+)\.json$", os.path.basename(round_path))
            if not m:
                continue
            round_num = int(m.group(1))

        guesses_dict = payload.get("guesses")
        if not isinstance(guesses_dict, dict):
            continue

        # Keys in JSON are strings; sort by numeric agent ID.
        sorted_agent_ids = sorted((int(k) for k in guesses_dict.keys()))
        guesses = [int(guesses_dict[str(agent_id)]) for agent_id in sorted_agent_ids]

        rounds_data.append([int(round_num)] + guesses)
        round_meta[int(round_num)] = {
            "group_metric_name": "sum" if payload.get("mode") == "sum" else "avg",
            "group_metric_value": payload.get("average"),
            "rounded_group_value": payload.get("rounded_average"),
            "feedback": payload.get("feedback"),
            "parsing_failures_count": len(payload.get("parsing_failures") or {}),
            "fallback_responses_count": len(payload.get("fallback_responses") or {}),
        }

    rounds_data.sort(key=lambda x: x[0])
    return rounds_data, round_meta


def load_nearest_config(run_dir, base_dir):
    """Load config from run dir or its parent folders if available."""
    candidates = []
    current = run_dir

    # Walk from run directory up to base directory.
    while True:
        candidates.append(os.path.join(current, "config.json"))
        candidates.append(os.path.join(current, "experiment_config.json"))

        if os.path.abspath(current) == os.path.abspath(base_dir):
            break

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for config_path in candidates:
        if os.path.exists(config_path):
            loaded = load_json_file(config_path)
            if loaded is not None:
                return loaded, config_path

    return None, None


def find_game_logs(base_dir):
    """Find all game_log.txt files under base_dir recursively."""
    game_logs = []
    for root, _, files in os.walk(base_dir):
        if "game_log.txt" in files:
            game_logs.append(os.path.join(root, "game_log.txt"))
    return sorted(game_logs)


def find_run_dirs(base_dir):
    """Find all run directories that look like game outputs."""
    run_dirs = set()

    for game_log in find_game_logs(base_dir):
        run_dirs.add(os.path.dirname(game_log))

    for root, _, files in os.walk(base_dir):
        has_round_json = any(re.match(r"round_\d+\.json$", name) for name in files)
        if has_round_json:
            run_dirs.add(root)

    return sorted(run_dirs)


def sanitize_filename_part(value):
    """Sanitize string for safe filename use."""
    text = str(value)
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def build_output_stem(base_dir, run_dir, config):
    """Create a deterministic output filename stem for a run."""
    rel_run_dir = sanitize_filename_part(os.path.relpath(run_dir, base_dir))

    if config and isinstance(config, dict):
        mode = config.get("mode")
        temp = config.get("temperature")
        agents = config.get("agents", [])
        num_agents = len(agents) if isinstance(agents, list) else None
        model = None
        if isinstance(agents, list) and agents:
            model = agents[0].get("model")

        parts = [rel_run_dir]
        if mode is not None:
            parts.append(str(mode))
        if num_agents is not None:
            parts.append(f"a{num_agents}")
        if temp is not None:
            parts.append(f"t{temp}")
        if model:
            parts.append(sanitize_filename_part(model))
        return "_".join(parts)

    return rel_run_dir


def unique_output_path(output_dir, stem, success_type):
    """Avoid overwriting by adding an index suffix when needed."""
    base_name = f"{stem}_{success_type}.csv"
    output_path = os.path.join(output_dir, base_name)

    if not os.path.exists(output_path):
        return output_path

    idx = 2
    while True:
        output_path = os.path.join(output_dir, f"{stem}_{success_type}_{idx}.csv")
        if not os.path.exists(output_path):
            return output_path
        idx += 1


def save_to_csv(rounds_data, filename, run_label, success_type):
    """Save rounds data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        num_agents = len(rounds_data[0]) - 1 if rounds_data else 10
        header = ['round'] + [f'agent_{i+1}' for i in range(num_agents)]
        writer.writerow(header)

        # Data rows
        writer.writerows(rounds_data)

    print(f"Saved: {filename} ({len(rounds_data)} rounds) - {run_label} ({success_type})")


def determine_success_type(summary, game_log_content):
    """Determine SUCCESS/FAILURE using summary first, then log content."""
    if isinstance(summary, dict) and "solved" in summary:
        return "SUCCESS" if bool(summary.get("solved")) else "FAILURE"

    if game_log_content and "CORRECT" in game_log_content:
        return "SUCCESS"

    return "FAILURE"


def build_run_metadata(base_dir, run_dir, config, summary, config_path):
    """Build consistent metadata fields for consolidated output."""
    rel_run_dir = os.path.relpath(run_dir, base_dir)
    run_name = os.path.basename(run_dir)

    agents = config.get("agents", []) if isinstance(config, dict) else []
    model = None
    if isinstance(agents, list) and agents:
        model = agents[0].get("model")
    if model is None and isinstance(config, dict):
        model = config.get("model")

    mode = config.get("mode") if isinstance(config, dict) else None
    temperature = config.get("temperature") if isinstance(config, dict) else None
    num_agents = len(agents) if isinstance(agents, list) else None
    mystery_number = None

    if isinstance(summary, dict):
        mystery_number = summary.get("mystery_number")
    if mystery_number is None and isinstance(config, dict):
        mystery_number = config.get("mystery_number")

    run_id = None
    run_id_match = re.search(r"run_(\d+)$", run_name)
    if run_id_match:
        run_id = int(run_id_match.group(1))

    experiment_folder = rel_run_dir.split(os.sep)[0] if os.sep in rel_run_dir else rel_run_dir

    return {
        "run_label": rel_run_dir,
        "run_name": run_name,
        "experiment_folder": experiment_folder,
        "mode": mode,
        "temperature": temperature,
        "num_agents": num_agents,
        "model": model,
        "mystery_number": mystery_number,
        "solved": summary.get("solved") if isinstance(summary, dict) else None,
        "run_id": run_id,
        "config_path": os.path.relpath(config_path, base_dir) if config_path else None,
    }


def append_long_rows(long_rows, rounds_data, run_meta, success_type, round_meta):
    """Append per-agent long-format rows for statistical analysis."""
    for row in rounds_data:
        round_num = row[0]
        guesses = row[1:]
        this_round_meta = round_meta.get(round_num, {}) if isinstance(round_meta, dict) else {}

        for idx, guess in enumerate(guesses):
            long_rows.append(
                {
                    "run_label": run_meta["run_label"],
                    "run_name": run_meta["run_name"],
                    "experiment_folder": run_meta["experiment_folder"],
                    "run_id": run_meta["run_id"],
                    "round": round_num,
                    "agent_id": idx,
                    "guess": guess,
                    "mode": run_meta["mode"],
                    "temperature": run_meta["temperature"],
                    "num_agents": run_meta["num_agents"] if run_meta["num_agents"] is not None else len(guesses),
                    "model": run_meta["model"],
                    "mystery_number": run_meta["mystery_number"],
                    "group_metric_name": this_round_meta.get("group_metric_name"),
                    "group_metric_value": this_round_meta.get("group_metric_value"),
                    "rounded_group_value": this_round_meta.get("rounded_group_value"),
                    "feedback": this_round_meta.get("feedback"),
                    "parsing_failures_count": this_round_meta.get("parsing_failures_count"),
                    "fallback_responses_count": this_round_meta.get("fallback_responses_count"),
                    "success_type": success_type,
                    "solved": run_meta["solved"],
                    "config_path": run_meta["config_path"],
                }
            )


def save_long_csv(rows, path):
    """Save consolidated long-format CSV for emergence analysis."""
    columns = [
        "run_label",
        "run_name",
        "experiment_folder",
        "run_id",
        "round",
        "agent_id",
        "guess",
        "mode",
        "temperature",
        "num_agents",
        "model",
        "mystery_number",
        "group_metric_name",
        "group_metric_value",
        "rounded_group_value",
        "feedback",
        "parsing_failures_count",
        "fallback_responses_count",
        "success_type",
        "solved",
        "config_path",
    ]

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved consolidated analysis CSV: {path} ({len(rows)} rows)")


def extract_game_data(base_dir, output_dir):
    """Extract game data from all runs under base_dir and save as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    run_dirs = find_run_dirs(base_dir)
    if not run_dirs:
        print(f"No run directories found in {base_dir}")
        return

    print(f"Found {len(run_dirs)} run directory(ies) under {base_dir}")

    total_saved = 0
    total_skipped = 0
    long_rows = []

    for run_dir in run_dirs:
        run_label = os.path.relpath(run_dir, base_dir)
        print(f"\nProcessing run: {run_label}")

        config, config_path = load_nearest_config(run_dir, base_dir)
        if config_path:
            print(f"Using config: {os.path.relpath(config_path, base_dir)}")
        else:
            print("Warning: no config.json or experiment_config.json found nearby")

        summary = load_json_file(os.path.join(run_dir, "summary.json"))

        rounds_data, round_meta = extract_round_data_from_json(run_dir)

        game_log_path = os.path.join(run_dir, "game_log.txt")
        game_log_content = None
        if os.path.exists(game_log_path):
            try:
                with open(game_log_path, "r") as f:
                    game_log_content = f.read()
            except Exception as e:
                print(f"Warning: error reading {game_log_path}: {e}")

        if not rounds_data and game_log_content:
            rounds_data = extract_round_data_from_log(game_log_content)
            round_meta = extract_round_metadata_from_log(game_log_content)

        if not rounds_data:
            print("No round data found; skipping")
            total_skipped += 1
            continue

        success_type = determine_success_type(summary, game_log_content)

        stem = build_output_stem(base_dir, run_dir, config)
        csv_path = unique_output_path(output_dir, stem, success_type)
        save_to_csv(rounds_data, csv_path, run_label, success_type)

        run_meta = build_run_metadata(base_dir, run_dir, config or {}, summary or {}, config_path)
        append_long_rows(long_rows, rounds_data, run_meta, success_type, round_meta)

        total_saved += 1

    if long_rows:
        consolidated_path = os.path.join(output_dir, "all_runs_emergence_long.csv")
        save_long_csv(long_rows, consolidated_path)

    print(f"\nDone. Saved {total_saved} CSV file(s). Skipped {total_skipped} run(s).")


def main():
    parser = argparse.ArgumentParser(description="Extract game logs to CSV files.")
    parser.add_argument(
        "--base-dir",
        default="./results",
        help="Directory containing experiment results (default: ./results)",
    )
    parser.add_argument(
        "--output-dir",
        default="./sampled_data",
        help="Directory to write CSV files (default: ./sampled_data)",
    )

    args = parser.parse_args()
    extract_game_data(args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()