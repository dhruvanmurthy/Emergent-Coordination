import os
import re
import csv
import json

def extract_round_data(content):
    """Extract round data from game log content."""
    rounds_data = []
    pattern = r"Round (\d+): guesses=\[(.*?)\]"
    
    for match in re.finditer(pattern, content):
        round_num = int(match.group(1))
        guesses_str = match.group(2)
        guesses = [int(x.strip()) for x in guesses_str.split(',')]
        
        # Create row: [round_number, guess1, guess2, ...]
        row = [round_num] + guesses
        rounds_data.append(row)
    
    # Sort by round number
    rounds_data.sort(key=lambda x: x[0])
    return rounds_data

def load_experiment_config(exp_dir):
    """Load experiment_config.json from the experiment directory."""
    config_path = os.path.join(exp_dir, "experiment_config.json")
    if not os.path.exists(config_path):
        print(f"Warning: config not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        return json.load(f)


def save_to_csv(rounds_data, filename, exp_name, config_dir, run_name, success_type):
    """Save rounds data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        num_agents = len(rounds_data[0]) - 1 if rounds_data else 10
        header = ['round'] + [f'agent_{i+1}' for i in range(num_agents)]
        writer.writerow(header)

        # Data rows
        writer.writerows(rounds_data)

    print(f"Saved: {filename} ({len(rounds_data)} rounds) - {exp_name}/{config_dir}/{run_name} ({success_type})")


def extract_game_data(base_dir, experiment_names, max_runs=20):
    """Extract game data from successful and unsuccessful experiments and save as CSV files."""

    for exp_name in experiment_names:
        exp_dir = os.path.join(base_dir, exp_name)
        print(f"\nProcessing experiment: {exp_name}")

        config = load_experiment_config(exp_dir)
        if config is None:
            continue

        agents_list = config.get('agents_list', [])
        temp_list = config.get('temp_list', [])

        if not agents_list or not temp_list:
            print(f"Missing agents_list or temp_list in {exp_name}/experiment_config.json")
            continue

        for agents in agents_list:
            for temp in temp_list:
                config_dir = f"sum_a{agents}_t{temp:.1f}"
                found_success = False
                found_failure = False
                run_num = 1

                # Search through runs until we find both types or exhaust runs
                while (not found_success or not found_failure) and run_num <= max_runs:
                    run_name = f"run_{run_num:03d}"
                    game_log_path = os.path.join(exp_dir, config_dir, run_name, "game_log.txt")

                    if not os.path.exists(game_log_path):
                        run_num += 1
                        continue

                    # Read the game log
                    try:
                        with open(game_log_path, 'r') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading {game_log_path}: {e}")
                        run_num += 1
                        continue

                    # Extract round data
                    rounds_data = extract_round_data(content)

                    if not rounds_data:
                        print(f"No round data found in {exp_name}/{config_dir}/{run_name}")
                        run_num += 1
                        continue

                    # Check if experiment was successful
                    is_successful = "CORRECT" in content
                    output_name = exp_name.replace('massive_experiment_', '')
                    output_name = f"{output_name}_{config_dir}"

                    if is_successful and not found_success:
                        # Save successful run
                        csv_filename = f"{output_name}_SUCCESS.csv"
                        save_to_csv(rounds_data, csv_filename, exp_name, config_dir, run_name, "SUCCESS")
                        found_success = True

                    elif not is_successful and not found_failure:
                        # Save unsuccessful run
                        csv_filename = f"{output_name}_FAILURE.csv"
                        save_to_csv(rounds_data, csv_filename, exp_name, config_dir, run_name, "FAILURE")
                        found_failure = True

                    run_num += 1

                if not found_success:
                    print(f"No successful runs found for {exp_name}/{config_dir}")
                if not found_failure:
                    print(f"No unsuccessful runs found for {exp_name}/{config_dir}")

# Usage
experiment_names = [
    "massive_experiment_openrouter_free_20260704_163043"
]

# Set your base directory path here
base_directory = "./results"  # Current directory, adjust as needed

extract_game_data(base_directory, experiment_names)