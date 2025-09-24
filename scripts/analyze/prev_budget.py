import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean
import matplotlib.pyplot as plt

target_attempt = 300

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_name = "h100_level_3-metr_trial_0"
root_dir = (curr_dir / "../../data/runs" / run_name / "levels/level_3-metr").resolve()

eager_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/eager.json").resolve()
sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/prev_agents_300/best_solutions").resolve()
output_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/prev_agents.json").resolve()
plot_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/figs/convergence/prev_convergence.pdf").resolve()

OUTPUT_SOLUTIONS = False

# Blacklist: tuples of (task_number, phase_num, agent_num, step_num)
BLACKLIST = {
    "h100_level_3-metr_trial_0": {
        (40, 4, 297, 1)
    }
}


# Load eager runtimes (with safe fallback)
try:
    with open(eager_path) as f:
        eager_runtimes = json.load(f)
except FileNotFoundError:
    print(f"Warning: eager runtimes file not found at {eager_path}. Continuing with empty eager runtimes (speedups will be skipped).")
    eager_runtimes = {"results": []}
except Exception as e:
    print(f"Warning: failed to load eager runtimes ({e}). Continuing with empty eager runtimes (speedups will be skipped).")
    eager_runtimes = {"results": []}

# Ensure eager_runtimes has expected structure
if not isinstance(eager_runtimes, dict) or "results" not in eager_runtimes or not isinstance(eager_runtimes["results"], list):
    print("Warning: eager runtimes file has unexpected format. Expected JSON object with a 'results' list. Continuing with empty results.")
    eager_runtimes = {"results": []}


def get_runtime(data, task_number: int):
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


def get_runtime_progress_and_best(task_path, task_number, target_attempt):
    """
    Walk steps in sorted numeric order across all phases/agents,
    track the best runtime seen so far (ignoring blacklisted combos).
    Stop at target_attempt.

    Returns:
        - A list of best-so-far runtimes for each step up to the target.
        - The final best runtime found within the target attempts.
        - The path to the kernel.py of the final best runtime.
        - The (phase_num, agent_num, step_num) triple that produced it.
    """
    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None
    best_combo = None  # (phase_num, agent_num, step_num)
    progress_list = []

    phases_root = os.path.join(task_path, "phases")
    if not os.path.exists(phases_root):
        return [None] * target_attempt, None, None, None

    phase_names = [
        d for d in os.listdir(phases_root)
        if os.path.isdir(os.path.join(phases_root, d)) and d.startswith("phase_")
    ]
    phase_names.sort(key=lambda x: numeric_suffix(x, "phase_"))

    stop_processing = False
    for phase_name in phase_names:
        phase_num = numeric_suffix(phase_name, "phase_")
        phase_path = os.path.join(phases_root, phase_name)
        agents_root = os.path.join(phase_path, "agents")
        if not os.path.exists(agents_root):
            continue

        agent_names = [
            d for d in os.listdir(agents_root)
            if os.path.isdir(os.path.join(agents_root, d)) and d.startswith("agent_")
        ]
        agent_names.sort(key=lambda x: numeric_suffix(x, "agent_"))

        for agent_name in agent_names:
            agent_num = numeric_suffix(agent_name, "agent_")
            agent_path = os.path.join(agents_root, agent_name)
            step_names = [
                d for d in os.listdir(agent_path)
                if os.path.isdir(os.path.join(agent_path, d)) and d.startswith("step_")
            ]
            step_names.sort(key=lambda x: numeric_suffix(x, "step_"))

            for step_name in step_names:
                step_num = numeric_suffix(step_name, "step_")
                cumulative += 1

                # Skip if in blacklist
                if (task_number, phase_num, agent_num, step_num) in BLACKLIST.get(run_name, set()):
                    pass
                else:
                    step_path = os.path.join(agent_path, step_name)
                    eval_file = os.path.join(step_path, "eval_results.json")
                    code_file = os.path.join(step_path, "kernel.py")

                    if os.path.exists(eval_file):
                        try:
                            with open(eval_file, "r") as f:
                                eval_data = json.load(f)
                            runtime = eval_data.get("eval_results", {}).get("runtime")
                            if runtime is not None and runtime < best_runtime:
                                best_runtime = runtime
                                best_code_path = code_file if os.path.exists(code_file) else None
                                best_combo = (phase_num, agent_num, step_num)
                        except Exception:
                            # ignore malformed eval files
                            pass

                progress_list.append(None if best_runtime == float("inf") else best_runtime)

                if cumulative >= target_attempt:
                    stop_processing = True
                    break
            if stop_processing:
                break
        if stop_processing:
            break

    # Pad the progress list if the total number of steps was less than the target
    final_best_for_padding = None if best_runtime == float("inf") else best_runtime
    while len(progress_list) < target_attempt:
        progress_list.append(final_best_for_padding)

    final_best_runtime = None if best_runtime == float("inf") else best_runtime
    return progress_list, final_best_runtime, best_code_path, best_combo


if __name__ == "__main__":
    sol_dest_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect tasks
    task_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("task")
    ]
    task_names.sort(key=lambda x: numeric_suffix(x, "task"))

    results = []
    speedup_list = []
    all_speedups_progress = [] # For convergence plot

    for task_name in task_names:
        task_number = numeric_suffix(task_name, "task")
        task_path = os.path.join(root_dir, task_name)
        
        progress, best, best_code_path, best_combo = get_runtime_progress_and_best(
            task_path, task_number, target_attempt
        )
        eager = get_runtime(eager_runtimes, task_number)

        # --- Process Final Result ---
        if best is not None:
            results.append({
                "problem_id": task_number,
                "runtime": best
            })

            combo_str = (f"phase_{best_combo[0]}/agent_{best_combo[1]}/step_{best_combo[2]}"
                         if best_combo else "N/A")

            if eager is not None and best > 0:
                speedup = eager / best
                speedup_list.append(speedup)
                print(f"{task_name} (id={task_number}): best={best:.6f}, "
                      f"eager={eager:.6f}, speedup={speedup:.3f}, at {combo_str}")
            else:
                print(f"{task_name} (id={task_number}): best={best:.6f}, "
                      f"eager=N/A, speedup=N/A, at {combo_str}")

            if best_code_path:
                dest_file = sol_dest_dir / f"task_{task_number}.py"
                if OUTPUT_SOLUTIONS:
                    shutil.copy(best_code_path, dest_file)
        else:
            print(f"{task_name} (id={task_number}): missing runtime")

        # --- Process Progress Data (for convergence plot) ---
        if eager is not None:
            speedup_progress = []
            for r in progress:
                # If no valid runtime, assume neutral speedup of 1.0
                # Clamp slowdowns to 1.0 as well for a cleaner plot
                if r is None or r <= 0:
                    speedup_progress.append(1.0)
                else:
                    s = eager / r
                    speedup_progress.append(max(1.0, s))
            all_speedups_progress.append(speedup_progress)

    # --- Finalize and Save Results ---

    # 1. Write JSON in requested format
    output_data = {
        "title": "Ours (prev, agents)",
        "results": sorted(results, key=lambda x: x["problem_id"])
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nRuntimes written to {output_path}")

    # 2. Print geometric mean of speedups
    if speedup_list:
        geo_mean = gmean(speedup_list)
        print(f"\nGeometric mean speedup across {len(speedup_list)} tasks = {geo_mean:.3f}")

    # 3. Generate and save convergence plot
    if all_speedups_progress:
        # Transpose list to calculate gmean for each step index
        geomean_curve = []
        for step_idx in range(target_attempt):
            vals_at_step = [
                task_s[step_idx] for task_s in all_speedups_progress
            ]
            geomean_curve.append(gmean(vals_at_step))

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, target_attempt + 1), geomean_curve)
        plt.xlabel("Cumulative Step Number")
        plt.ylabel("Speedup over PyTorch Eager (Geomean)")
        plt.title("Previous Agents Speedup by Step (Level 3-metr, H100)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Convergence plot saved to {plot_path}\n")
