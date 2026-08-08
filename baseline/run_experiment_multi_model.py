# Updated run_experiment.py to handle multiple models via command line
from experiment import GameMaster, ParsingError  # Import ParsingError
from datetime import datetime
import os
import asyncio
import json
import time
from settings import get_optional, get_required, require_keys


async def run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds):
    """Run a single configuration asynchronously"""
    config_id = f"{agents}a_t{temp:.1f}_r{run_id}"
    print(f"    🚀 Starting: {config_id}")
    
    try:
        game = GameMaster(mode=mode, temperature=temp, max_rounds=max_rounds, 
                         num_agents=agents, batch_folder=batch_folder, run_id=run_id)
        for i in range(agents):
            game.add_agent(model)
            # game.add_agent(model, client_type)
        
        await game.play_game()
        
        print(f"    ✅ Completed: {config_id}")
        return {"status": "success", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id}
        
    except ParsingError as e:
        print(f"    🔤 Parsing Failed: {config_id} - {str(e)[:100]}")
        return {"status": "parsing_failed", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "error": str(e)}
        
    except Exception as e:
        print(f"    ❌ Failed: {config_id} - {str(e)[:100]}")
        return create_fallback_result(config_id, agents, temp, run_id, str(e))

def create_fallback_result(config_id, agents, temp, run_id, error_msg):
    """Create a fallback result when game completely fails"""
    return {
        "status": "failed_with_fallback",
        "config": config_id,
        "agents": agents,
        "temp": temp,
        "run_id": run_id,
        "error": error_msg,
        "fallback_data": {
            "total_rounds": 0,
            "solved": False,
            "completed_successfully": False,
            "note": "Game failed completely, using fallback data"
        }
    }

def create_config_batches(agents_list, temp_list, runs_per_config, batch_size=20):
    """Create batches of configurations to run"""
    all_configs = []
    
    for agents in agents_list:
        for temp in temp_list:
            for run_id in range(1, runs_per_config + 1):
                all_configs.append((agents, temp, run_id))
    
    # Split into batches
    batches = []
    for i in range(0, len(all_configs), batch_size):
        batches.append(all_configs[i:i + batch_size])
    
    return batches

def save_progress(batch_folder, batch_num, batch_results, total_batches):
    """Save progress to file"""
    progress_file = os.path.join(batch_folder, "progress.json")
    
    # Load existing progress
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {"batches_completed": [], "total_batches": total_batches, "start_time": time.time()}
    
    # Add current batch results
    batch_summary = {
        "batch_num": batch_num,
        "total_configs": len(batch_results),
        "successful": sum(1 for r in batch_results if r["status"] == "success"),
        "failed": sum(1 for r in batch_results if r["status"] == "failed"),
        "parsing_failed": sum(1 for r in batch_results if r["status"] == "parsing_failed"),
        "failed_with_fallback": sum(1 for r in batch_results if r["status"] == "failed_with_fallback"),
        "completion_time": time.time(),
        "results": batch_results
    }
    
    progress["batches_completed"].append(batch_summary)
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    return progress

def save_failure_summary(batch_folder, progress):
    """Save summary of all failures for analysis"""
    parsing_failures = []
    api_failures = []
    game_failures = []
    
    for batch in progress["batches_completed"]:
        for result in batch["results"]:
            if result["status"] == "parsing_failed":
                parsing_failures.append({
                    "batch_num": batch["batch_num"],
                    "config": result["config"],
                    "agents": result["agents"],
                    "temp": result["temp"],
                    "run_id": result["run_id"],
                    "error": result["error"]
                })
            elif result["status"] == "failed_with_fallback":
                game_failures.append({
                    "batch_num": batch["batch_num"],
                    "config": result["config"],
                    "agents": result["agents"],
                    "temp": result["temp"],
                    "run_id": result["run_id"],
                    "error": result["error"]
                })
    
    total_configs = sum(batch["total_configs"] for batch in progress["batches_completed"])
    
    failure_summary = {
        "total_parsing_failures": len(parsing_failures),
        "total_game_failures": len(game_failures),
        "parsing_failure_rate": len(parsing_failures) / total_configs if total_configs > 0 else 0,
        "game_failure_rate": len(game_failures) / total_configs if total_configs > 0 else 0,
        "total_failure_rate": (len(parsing_failures) + len(game_failures)) / total_configs if total_configs > 0 else 0,
        "failures_by_agent_count": {},
        "failures_by_temperature": {},
        "detailed_parsing_failures": parsing_failures,
        "detailed_game_failures": game_failures
    }
    
    # Analyze patterns
    for failure in parsing_failures:
        # By agent count
        agents = failure["agents"]
        if agents not in failure_summary["failures_by_agent_count"]:
            failure_summary["failures_by_agent_count"][agents] = 0
        failure_summary["failures_by_agent_count"][agents] += 1
        
        # By temperature
        temp = failure["temp"]
        if temp not in failure_summary["failures_by_temperature"]:
            failure_summary["failures_by_temperature"][temp] = 0
        failure_summary["failures_by_temperature"][temp] += 1
    
    # Save to file
    with open(os.path.join(batch_folder, "parsing_failures_analysis.json"), 'w') as f:
        json.dump(failure_summary, f, indent=2)
    
    return failure_summary

def load_progress(batch_folder):
    """Load existing progress if available"""
    progress_file = os.path.join(batch_folder, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

async def run_batch(batch_configs, batch_num, total_batches, batch_folder, model, client_type, mode, max_rounds, max_concurrent=10):
    """Run a single batch of configurations"""
    batch_start_time = time.time()
    
    print(f"\n📦 BATCH {batch_num}/{total_batches}")
    print(f"🔢 Configs in this batch: {len(batch_configs)}")
    print(f"⚡ Max concurrent: {max_concurrent}")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_run_config(config):
        async with semaphore:
            agents, temp, run_id = config
            return await run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds)
    
    # Run all configs in this batch
    batch_results = await asyncio.gather(
        *[limited_run_config(config) for config in batch_configs],
        return_exceptions=True
    )
    
    # Handle any exceptions
    processed_results = []
    for result in batch_results:
        if isinstance(result, Exception):
            processed_results.append({
                "status": "failed", 
                "config": "unknown", 
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    # Calculate batch statistics
    successful = sum(1 for r in processed_results if r["status"] == "success")
    failed = sum(1 for r in processed_results if r["status"] == "failed")
    parsing_failed = sum(1 for r in processed_results if r["status"] == "parsing_failed")
    batch_duration = time.time() - batch_start_time
    
    print(f"  ✅ Successful: {successful}/{len(batch_configs)}")
    print(f"  ❌ Failed: {failed}/{len(batch_configs)}")
    print(f"  🔤 Parsing Failed: {parsing_failed}/{len(batch_configs)}")
    print(f"  ⏱️  Duration: {batch_duration:.1f}s")
    
    # Save progress
    progress = save_progress(batch_folder, batch_num, processed_results, total_batches)
    
    return processed_results


def load_runner_settings():
    section = "run_experiment_multi_model"
    require_keys(
        section,
        [
            "models",
            "client_type",
            "mode",
            "max_rounds",
            "agents_list",
            "temp_list",
            "runs_per_config",
        ],
    )

    models = get_required(section, "models")
    if not isinstance(models, list) or not models:
        raise ValueError("run_experiment_multi_model.models must be a non-empty list")

    return {
        "models": models,
        "client_type": get_required(section, "client_type"),
        "mode": get_required(section, "mode"),
        "max_rounds": get_required(section, "max_rounds"),
        "agents_list": get_required(section, "agents_list"),
        "temp_list": get_required(section, "temp_list"),
        "runs_per_config": get_required(section, "runs_per_config"),
        "batch_size": get_optional(section, "batch_size", 20),
        "max_concurrent": get_optional(section, "max_concurrent", 8),
        "resume_from_batch": get_optional(section, "resume_from_batch", None),
        "results_prefix": get_optional(section, "results_prefix", "massive_experiment"),
    }


async def run_for_model(model, cfg):
    client_type = cfg["client_type"]
    mode = cfg["mode"]
    max_rounds = cfg["max_rounds"]
    agents_list = cfg["agents_list"]
    temp_list = cfg["temp_list"]
    runs_per_config = cfg["runs_per_config"]
    batch_size = cfg["batch_size"]
    max_concurrent = cfg["max_concurrent"]
    resume_from_batch = cfg["resume_from_batch"]
    results_prefix = cfg["results_prefix"]

    total_configs = len(agents_list) * len(temp_list) * runs_per_config
    model_safe_name = model.replace("/", "_").replace(":", "_")

    print(f"🎯 MASSIVE EXPERIMENT SETUP:")
    print(f"   🤖 Model: {model}")
    print(f"   Agent counts: {len(agents_list)} ({min(agents_list)} to {max(agents_list)})")
    print(f"   Temperatures: {len(temp_list)} ({min(temp_list):.1f} to {max(temp_list):.1f})")
    print(f"   Runs per config: {runs_per_config}")
    print(f"   📊 TOTAL CONFIGS: {total_configs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   ⚡ Max concurrent per batch: {max_concurrent}")

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = f"results/{results_prefix}_{model_safe_name}_{batch_timestamp}"
    os.makedirs(batch_folder, exist_ok=True)

    experiment_config = {
        "model": model,
        "agents_list": agents_list,
        "temp_list": temp_list,
        "runs_per_config": runs_per_config,
        "total_configs": total_configs,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "client_type": client_type,
        "mode": mode,
        "max_rounds": max_rounds,
        "start_time": datetime.now().isoformat()
    }

    with open(os.path.join(batch_folder, "experiment_config.json"), 'w') as f:
        json.dump(experiment_config, f, indent=2)

    print(f"📁 Results folder: {batch_folder}")

    print(f"\n🔄 Creating batches...")
    config_batches = create_config_batches(agents_list, temp_list, runs_per_config, batch_size)
    total_batches = len(config_batches)
    print(f"📦 Created {total_batches} batches")

    existing_progress = load_progress(batch_folder)
    start_batch = 1

    if existing_progress and resume_from_batch:
        start_batch = resume_from_batch
        print(f"🔄 Resuming from batch {start_batch}")
    elif existing_progress:
        completed_batches = len(existing_progress["batches_completed"])
        if completed_batches < total_batches:
            start_batch = completed_batches + 1
            print(f"🔄 Found existing progress. Resuming from batch {start_batch}")
        else:
            print(f"✅ All batches already completed!")
            return

    experiment_start_time = time.time()

    for batch_num in range(start_batch, total_batches + 1):
        batch_configs = config_batches[batch_num - 1]

        try:
            await run_batch(
                batch_configs, batch_num, total_batches,
                batch_folder, model, client_type, mode, max_rounds, max_concurrent
            )

            elapsed_time = time.time() - experiment_start_time
            progress_pct = (batch_num / total_batches) * 100

            if batch_num < total_batches:
                estimated_total_time = elapsed_time * (total_batches / batch_num)
                remaining_time = estimated_total_time - elapsed_time
                print(f"🎯 Overall Progress: {batch_num}/{total_batches} batches ({progress_pct:.1f}%)")
                print(f"⏱️  Elapsed: {elapsed_time/60:.1f}m | Estimated remaining: {remaining_time/60:.1f}m")

        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted at batch {batch_num}")
            print(f"💾 Progress saved. Resume with: resume_from_batch = {batch_num}")
            break
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            print(f"💾 Progress saved. You can resume from this batch.")
            continue

    final_progress = load_progress(batch_folder)
    if final_progress:
        total_successful = sum(batch["successful"] for batch in final_progress["batches_completed"])
        total_failed = sum(batch["failed"] for batch in final_progress["batches_completed"])
        total_parsing_failed = sum(batch["parsing_failed"] for batch in final_progress["batches_completed"])
        total_duration = time.time() - final_progress["start_time"]

        failure_analysis = save_failure_summary(batch_folder, final_progress)

        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"🤖 Model: {model}")
        print(f"📊 Final Results:")
        print(f"   ✅ Successful configs: {total_successful}")
        print(f"   ❌ Failed configs: {total_failed}")
        print(f"   🔤 Parsing failed configs: {total_parsing_failed}")
        print(f"   📈 Parsing failure rate: {failure_analysis['parsing_failure_rate']:.1%}")
        print(f"   📦 Batches completed: {len(final_progress['batches_completed'])}/{total_batches}")
        print(f"   ⏱️  Total duration: {total_duration/3600:.1f} hours")
        print(f"📁 Results saved in: {batch_folder}")
        print(f"📋 Parsing analysis saved in: parsing_failures_analysis.json")

async def main():
    cfg = load_runner_settings()

    print(f"🧩 Loaded settings from settings.py for run_experiment_multi_model")
    for model in cfg["models"]:
        await run_for_model(model, cfg)

if __name__ == "__main__":
    import time
    total_start_time = time.time()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_end_time = time.time()
        total_elapsed = total_end_time - total_start_time
        print(f"\n🏁 Total program execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")