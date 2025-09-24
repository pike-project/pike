import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean
import matplotlib.pyplot as plt

# --- Configuration ---
target_attempt = 300
run_name = "h100_level_3-metr_trial_4"

# --- Path Setup ---
curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# Input paths
root_dir = (curr_dir / "../../../openevolve/examples/kernelbench/openevolve_output_lrc" / run_name / "tasks").resolve()
eager_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/eager.json").resolve()

# Output paths
sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/openevolve_pop_25_agents_300/best_solutions").resolve()
output_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/oe_agents.json").resolve()
plot_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/figs/convergence/oe_convergence.pdf").resolve()


# --- Helper Functions ---

def get_eager_runtime(data, task_number: int):
    """Return eager runtime for a given integer task id."""
    for v in data["results"]:
        if v["problem_id"] == task_number:
            return v["runtime"]
    return None


def numeric_suffix(name: str, prefix: str) -> int:
    """Extract integer suffix from names like 'task12', 'iter_34', 'attempt_5'."""
    try:
        return int(name.replace(prefix, ""))
    except ValueError:
        return -1


def get_runtime_progress_and_best(task_path, target_attempt):
    """
    Walk through attempts in sorted numeric order, tracking the best runtime seen so far.
    Stop at target_attempt.

    Returns:
        - A list of best-so-far runtimes for each attempt up to the target.
        - The final best runtime found within the target attempts.
        - The path to the code file of the final best runtime.
    """
    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        return [None] * target_attempt, None, None

    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None
    progress_list = []

    iter_names = [
        d for d in os.listdir(iter_output_dir)
        if os.path.isdir(os.path.join(iter_output_dir, d)) and d.startswith("iter_")
    ]
    iter_names.sort(key=lambda x: numeric_suffix(x, "iter_"))

    stop_processing = False
    for iter_name in iter_names:
        iter_path = os.path.join(iter_output_dir, iter_name)
        attempts_dir = os.path.join(iter_path, "attempts")
        if not os.path.exists(attempts_dir):
            continue

        attempt_names = [
            d for d in os.listdir(attempts_dir)
            if os.path.isdir(os.path.join(attempts_dir, d)) and d.startswith("attempt_")
        ]
        attempt_names.sort(key=lambda x: numeric_suffix(x, "attempt_"))

        for attempt_name in attempt_names:
            cumulative += 1
            attempt_path = os.path.join(attempts_dir, attempt_name)
            metrics_file = os.path.join(attempt_path, "metrics_artifacts.json")
            code_file = os.path.join(attempt_path, "code.py")

            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)
                    runtime = metrics_data.get("metrics", {}).get("runtime")
                    if runtime is not None and runtime < best_runtime:
                        best_runtime = runtime
                        if os.path.exists(code_file):
                            best_code_path = code_file
                except (json.JSONDecodeError, KeyError):
                    pass  # Ignore corrupted files or files with missing keys

            progress_list.append(None if best_runtime == float("inf") else best_runtime)

            if cumulative >= target_attempt:
                stop_processing = True
                break
        if stop_processing:
            break

    # Pad the progress list if the total number of attempts was less than the target
    final_best_for_padding = None if best_runtime == float("inf") else best_runtime
    while len(progress_list) < target_attempt:
        progress_list.append(final_best_for_padding)

    final_best_runtime = None if best_runtime == float("inf") else best_runtime
    return progress_list, final_best_runtime, best_code_path


if __name__ == "__main__":
    # Create output directories if they don't exist
    sol_dest_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Load baseline eager runtimes
    with open(eager_path) as f:
        eager_runtimes = json.load(f)

    # Collect and sort task directories
    task_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("task")
    ]
    task_names.sort(key=lambda x: numeric_suffix(x, "task"))

    # Initialize collectors for results
    results = []
    final_speedups = []
    all_speedups_progress = [] # For convergence plot

    for task_name in task_names:
        task_number = numeric_suffix(task_name, "task")
        task_path = os.path.join(root_dir, task_name)

        # Extract both progress and final best result in one pass
        progress, best_runtime, best_code_path = get_runtime_progress_and_best(
            task_path, target_attempt
        )
        eager = get_eager_runtime(eager_runtimes, task_number)

        # --- Process Final Result (for JSON output and code saving) ---
        if best_runtime is not None:
            results.append({"problem_id": task_number, "runtime": best_runtime})
            print(f"{task_name} (id={task_number}): best runtime = {best_runtime:.6f}")

            if eager is not None and best_runtime > 0:
                speedup = eager / best_runtime
                final_speedups.append(speedup)
                print(f"  eager={eager:.6f}, speedup={speedup:.3f}")
            else:
                print("  missing eager runtime, skipping speedup")

            if best_code_path:
                dest_file = sol_dest_dir / f"task_{task_number}.py"
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

    # 1. Write runtimes JSON file
    output_data = {
        "title": "Ours (OE, agents)",
        "results": sorted(results, key=lambda x: x["problem_id"])
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nRuntimes written to {output_path}")

    # 2. Print final geometric mean speedup
    if final_speedups:
        geo_mean = gmean(final_speedups)
        print(f"Geometric mean speedup across {len(final_speedups)} tasks = {geo_mean:.3f}")

    # 3. Generate and save convergence plot
    if all_speedups_progress:
        # Transpose list to calculate gmean for each attempt index
        geomean_curve = []
        for attempt_idx in range(target_attempt):
            vals_at_attempt = [
                task_s[attempt_idx] for task_s in all_speedups_progress
            ]
            geomean_curve.append(gmean(vals_at_attempt))

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, target_attempt + 1), geomean_curve)
        plt.xlabel("Attempt Number")
        plt.ylabel("Speedup over PyTorch Eager (Geomean)")
        plt.title("OpenEvolve Agents Speedup by Attempt (Level 3-metr, H100)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Convergence plot saved to {plot_path}\n")
