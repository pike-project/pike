import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean
import matplotlib.pyplot as plt
import pandas as pd

# --- Primary Configuration Flag ---
# Set to False for the original structure (phases/agents/steps)
# Set to True for the new structure (iter_output/iter/attempts)
USE_OPENEVOLVE_STRUCTURE = False

# --- Common Configuration ---
target_attempt = 300
OUTPUT_SOLUTIONS = False # Set to True to copy the best kernel/code files

# --- Structure-Specific Configurations ---
curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

if not USE_OPENEVOLVE_STRUCTURE:
    # CONFIGURATION FOR ORIGINAL "phases/agents/steps" STRUCTURE
    output_label = "prev_noagents"
    plot_title = "Previous No Agents Speedup by Step (Level 3-metr, H100)"
    plot_xlabel = "Cumulative Step Number"

    run_name = "h100_level_3-metr_prev_noagents_trial_0"
    root_dir = (curr_dir / "../../data/parallel_runs" / run_name / "runs/runs/run_0/run/levels/level_3-metr").resolve()
    # run_name = "h100_level_3-metr_trial_0"
    # root_dir = (curr_dir / "../../data/runs" / run_name / "levels/level_3-metr").resolve()
    eager_path = (curr_dir / "../../results/ours/h100_level3-metr/results/data/runtimes/eager.json").resolve()
    sol_dest_dir = (curr_dir / f"../../best_agent_solutions/h100/level3-metr/{output_label}_{target_attempt}/best_solutions").resolve()
    output_path = (curr_dir / f"../../results/ours/h100_level3-metr/results/data/runtimes/{output_label}.json").resolve()
    plot_path = (curr_dir / f"../../results/ours/h100_level3-metr/results/figs/convergence/{output_label}_convergence.pdf").resolve()
    all_trajectories_path = (curr_dir / f"../../results/ours/h100_level3-metr/results/data/tables/speedup_trajectories/{output_label}.csv").resolve()

    # PATCH 1: Updated comment to clarify behavior.
    # Blacklist format:
    # - Tuples (task_num, phase_num, agent_num, step_num) to blacklist a specific attempt.
    # - Integers (task_num) to set the speedup to 1.0 for an entire task.
    BLACKLIST = {
        "h100_level_3-metr_prev_noagents_trial_0": {
            39,
        },
        "h100_level_3-metr_trial_0": {
            (40, 4, 297, 1), # Blacklists a specific attempt
            # 42, # Sets speedup to 1.0 for a task
        },
    }
else:
    # CONFIGURATION FOR NEW "iter_output/iter/attempts" STRUCTURE
    run_name = "h100_level_3-metr_trial_4"
    output_label = "oe_agents"
    plot_title = "OpenEvolve Agents Speedup by Attempt (Level 3-metr, H100)"
    plot_xlabel = "Attempt Number"

    root_dir = (curr_dir / "../../../openevolve/examples/kernelbench/openevolve_output_lrc" / run_name / "tasks").resolve()
    eager_path = (curr_dir / "../../results/ours/h100_level3-metr/results/data/runtimes/eager.json").resolve()
    sol_dest_dir = (curr_dir / f"../../best_agent_solutions/h100/level3-metr/{output_label}_{target_attempt}/best_solutions").resolve()
    output_path = (curr_dir / f"../../results/ours/h100_level3-metr/results/data/runtimes/{output_label}.json").resolve()
    plot_path = (curr_dir / f"../../results/ours/h100_level3-metr/results/figs/convergence/{output_label}_convergence.pdf").resolve()
    all_trajectories_path = (curr_dir / f"../../results/ours/h100_level3-metr/results/data/tables/speedup_trajectories/{output_label}.csv").resolve()
    
    # PATCH 1: Updated comment to clarify behavior.
    # Blacklist format:
    # - Tuples (task_number, iter_num, attempt_num) to blacklist a specific attempt.
    # - Integers (task_number) to set the speedup to 1.0 for an entire task.
    BLACKLIST = {
        "h100_level_3-metr_trial_4": {
            # (13, 229, 0), # Blacklists a specific attempt
            13, # Sets speedup to 1.0 for a task
        },
    }


# --- Load Eager Runtimes (with robust error handling) ---
try:
    with open(eager_path) as f:
        eager_runtimes = json.load(f)
except FileNotFoundError:
    print(f"Warning: eager runtimes file not found at {eager_path}. Continuing with empty eager runtimes (speedups will be skipped).")
    eager_runtimes = {"results": []}
except Exception as e:
    print(f"Warning: failed to load eager runtimes ({e}). Continuing with empty eager runtimes (speedups will be skipped).")
    eager_runtimes = {"results": []}

if not isinstance(eager_runtimes, dict) or "results" not in eager_runtimes or not isinstance(eager_runtimes["results"], list):
    print("Warning: eager runtimes file has unexpected format. Expected JSON object with a 'results' list. Continuing with empty results.")
    eager_runtimes = {"results": []}


# --- Helper Functions ---

def get_eager_runtime(data, task_number: int):
    """Return eager runtime for given integer task id (or None if missing)."""
    for v in data.get("results", []):
        # tolerate either "problem_id" or "problem" keys if formats vary
        if v.get("problem_id") == task_number or v.get("problem") == task_number:
            return v.get("runtime")
    return None


def numeric_suffix(name: str, prefix: str) -> int:
    """Extract integer suffix from names like 'task12', 'phase_3', 'agent_4', 'step_7'."""
    try:
        return int(name.replace(prefix, "").replace("_", ""))
    except ValueError:
        raise Exception(f"Numeric suffix failed: name -> {name}, prefix -> {prefix}")


# --- Directory Traversal Logic ---

def get_progress_phases_agents_steps(task_path, task_number, target_attempt):
    """
    Traversal logic for the original structure: phases/agents/steps.
    Walks steps in sorted numeric order, tracks best runtime, ignoring blacklisted combos.
    Returns: (progress_list, best_runtime, best_code_path, best_combo_info)
    """
    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None
    best_combo = None  # (phase_num, agent_num, step_num)
    progress_list = []
    
    current_blacklist = BLACKLIST.get(run_name, set())

    phases_root = os.path.join(task_path, "phases")
    if not os.path.exists(phases_root):
        return [None] * target_attempt, None, None, None

    phase_names = sorted([d for d in os.listdir(phases_root) if d.startswith("phase_")], key=lambda x: numeric_suffix(x, "phase_"))

    stop_processing = False
    for phase_name in phase_names:
        phase_num = numeric_suffix(phase_name, "phase_")
        agents_root = os.path.join(phases_root, phase_name, "agents")
        if not os.path.exists(agents_root): continue

        agent_names = sorted([d for d in os.listdir(agents_root) if d.startswith("agent_")], key=lambda x: numeric_suffix(x, "agent_"))

        for agent_name in agent_names:
            agent_num = numeric_suffix(agent_name, "agent_")
            agent_path = os.path.join(agents_root, agent_name)
            step_names = sorted([d for d in os.listdir(agent_path) if d.startswith("step_")], key=lambda x: numeric_suffix(x, "step_"))

            for step_name in step_names:
                step_num = numeric_suffix(step_name, "step_")
                cumulative += 1

                if (task_number, phase_num, agent_num, step_num) not in current_blacklist:
                    eval_file = os.path.join(agent_path, step_name, "eval_results.json")
                    code_file = os.path.join(agent_path, step_name, "kernel.py")
                    if os.path.exists(eval_file):
                        try:
                            with open(eval_file, "r") as f: eval_data = json.load(f)
                            runtime = eval_data.get("eval_results", {}).get("runtime")
                            if runtime is not None and runtime < best_runtime:
                                best_runtime = runtime
                                best_code_path = code_file if os.path.exists(code_file) else None
                                best_combo = (phase_num, agent_num, step_num)
                        except Exception: pass # ignore malformed eval files

                progress_list.append(None if best_runtime == float("inf") else best_runtime)
                if cumulative >= target_attempt: stop_processing = True; break
            if stop_processing: break
        if stop_processing: break

    # Pad the progress list if needed
    final_best_for_padding = None if best_runtime == float("inf") else best_runtime
    while len(progress_list) < target_attempt:
        progress_list.append(final_best_for_padding)

    final_best_runtime = None if best_runtime == float("inf") else best_runtime
    return progress_list, final_best_runtime, best_code_path, best_combo


def get_progress_iters_attempts(task_path, task_number, target_attempt):
    """
    Traversal logic for the OpenEvolve structure: iter_output/iter/attempts.
    Walks attempts in sorted numeric order, tracking best runtime.
    Returns: (progress_list, best_runtime, best_code_path, best_combo_info)
    """
    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None
    best_combo = None # (iter_num, attempt_num)
    progress_list = []
    
    current_blacklist = BLACKLIST.get(run_name, set())

    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        return [None] * target_attempt, None, None, None
    
    iter_names = sorted([d for d in os.listdir(iter_output_dir) if d.startswith("iter_")], key=lambda x: numeric_suffix(x, "iter_"))
    
    stop_processing = False
    for iter_name in iter_names:
        iter_num = numeric_suffix(iter_name, "iter_")
        attempts_dir = os.path.join(iter_output_dir, iter_name, "attempts")
        if not os.path.exists(attempts_dir): continue

        attempt_names = sorted([d for d in os.listdir(attempts_dir) if d.startswith("attempt_")], key=lambda x: numeric_suffix(x, "attempt_"))
        
        for attempt_name in attempt_names:
            attempt_num = numeric_suffix(attempt_name, "attempt_")
            cumulative += 1
            
            if (task_number, iter_num, attempt_num) not in current_blacklist:
                metrics_file = os.path.join(attempts_dir, attempt_name, "metrics_artifacts.json")
                code_file = os.path.join(attempts_dir, attempt_name, "code.py")
                
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r") as f: metrics_data = json.load(f)
                        runtime = metrics_data.get("metrics", {}).get("runtime")
                        if runtime is not None and runtime < best_runtime:
                            best_runtime = runtime
                            best_code_path = code_file if os.path.exists(code_file) else None
                            best_combo = (iter_num, attempt_num)
                    except Exception: pass # Ignore corrupted files or files with missing keys
            
            progress_list.append(None if best_runtime == float("inf") else best_runtime)
            if cumulative >= target_attempt: stop_processing = True; break
        if stop_processing: break

    # Pad the progress list if needed
    final_best_for_padding = None if best_runtime == float("inf") else best_runtime
    while len(progress_list) < target_attempt:
        progress_list.append(final_best_for_padding)
    
    final_best_runtime = None if best_runtime == float("inf") else best_runtime
    return progress_list, final_best_runtime, best_code_path, best_combo


# --- Main Execution Logic ---

if __name__ == "__main__":
    sol_dest_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    task_names = sorted([d for d in os.listdir(root_dir) if d.startswith("task")], key=lambda x: numeric_suffix(x, "task"))

    results = []
    speedup_list = []
    all_speedups_progress = []
    included_task_names_for_csv = []
    
    current_blacklist = BLACKLIST.get(run_name, set())

    print(f"Processing run: {run_name} (Structure: {'OpenEvolve' if USE_OPENEVOLVE_STRUCTURE else 'Original'})")

    for task_name in task_names:
        task_number = numeric_suffix(task_name, "task")
        task_path = os.path.join(root_dir, task_name)
        
        is_task_speedup_blacklisted = task_number in current_blacklist

        if not USE_OPENEVOLVE_STRUCTURE:
            progress, best, best_code_path, best_combo = get_progress_phases_agents_steps(
                task_path, task_number, target_attempt
            )
        else:
            progress, best, best_code_path, best_combo = get_progress_iters_attempts(
                task_path, task_number, target_attempt
            )

        eager = get_eager_runtime(eager_runtimes, task_number)

        # --- Process Final Result ---
        if best is not None:
            results.append({"problem_id": task_number, "runtime": best})

            if best_combo:
                if not USE_OPENEVOLVE_STRUCTURE:
                    combo_str = f"phase_{best_combo[0]}/agent_{best_combo[1]}/step_{best_combo[2]}"
                else:
                    combo_str = f"iter_{best_combo[0]}/attempt_{best_combo[1]}"
            else:
                combo_str = "N/A"

            # PATCH 2: Calculate speedup, setting to 1.0 if blacklisted
            speedup = None
            if eager is not None:
                if is_task_speedup_blacklisted:
                    speedup = 1.0
                elif best > 0:
                    speedup = eager / best

            if speedup is not None:
                speedup_list.append(speedup)
                if is_task_speedup_blacklisted:
                    print(f"{task_name} (id={task_number}): best={best:.6f}, eager=N/A, speedup={speedup:.3f} (task blacklisted), at {combo_str}")
                else:
                    print(f"{task_name} (id={task_number}): best={best:.6f}, eager={eager:.6f}, speedup={speedup:.3f}, at {combo_str}")
            else:
                print(f"{task_name} (id={task_number}): best={best:.6f}, eager=N/A, speedup=N/A, at {combo_str}")

            if OUTPUT_SOLUTIONS and best_code_path:
                dest_file = sol_dest_dir / f"task_{task_number}.py"
                shutil.copy(best_code_path, dest_file)
        else:
            print(f"{task_name} (id={task_number}): missing runtime")

        # --- Process Progress Data (for convergence plot and CSV) ---
        # PATCH 3: Create speedup trajectory, setting to 1.0 if blacklisted
        if eager is not None:
            if is_task_speedup_blacklisted:
                speedup_progress = [1.0] * target_attempt
            else:
                speedup_progress = [max(1.0, eager / r) if r is not None and r > 0 else 1.0 for r in progress]
            all_speedups_progress.append(speedup_progress)
            task_id = numeric_suffix(task_name, "task")
            included_task_names_for_csv.append(f"task_{task_id}")


    # --- Finalize and Save Results ---

    # 1. Write JSON runtimes
    output_data = {"title": output_label, "results": sorted(results, key=lambda x: x["problem_id"])}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nRuntimes written to {output_path}")

    # 2. Print geometric mean of speedups
    if speedup_list:
        geo_mean = gmean(speedup_list)
        print(f"\nGeometric mean speedup across {len(speedup_list)} tasks = {geo_mean:.3f}")

    # 3. Generate and save the all_trajectories CSV
    if all_speedups_progress and included_task_names_for_csv:
        # PATCH 4: Use the curated list of task names to ensure columns match data
        df = pd.DataFrame(dict(zip(included_task_names_for_csv, all_speedups_progress)))
        df.index = pd.RangeIndex(start=1, stop=target_attempt + 1, name="attempt")
        all_trajectories_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(all_trajectories_path)
        print(f"Speedup trajectories saved to {all_trajectories_path}")

    # 4. Generate and save convergence plot
    if all_speedups_progress:
        geomean_curve = [gmean([s[i] for s in all_speedups_progress]) for i in range(target_attempt)]
        
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, target_attempt + 1), geomean_curve)
        plt.xlabel(plot_xlabel)
        plt.ylabel("Speedup over PyTorch Eager (Geomean)")
        plt.title(plot_title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Convergence plot saved to {plot_path}\n")
