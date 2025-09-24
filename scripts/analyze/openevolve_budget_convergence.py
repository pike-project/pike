import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean
import matplotlib.pyplot as plt

# Root directory
root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"
target_attempt = 300  # number of attempts

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
eager_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/eager.json").resolve()
sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/openevolve_pop_25_agents_300/best_solutions").resolve()
output_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/oe_agents.json").resolve()
plot_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/figs/convergence.pdf").resolve()

with open(eager_path) as f:
    eager_runtimes = json.load(f)


def get_runtime(data, task_number: int):
    """Return eager runtime for given integer task id."""
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


def best_runtimes_progress(task_path, target_attempt):
    """
    Return list of best-so-far runtimes for each attempt index up to target_attempt.
    """
    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        return [None] * target_attempt

    cumulative = 0
    best_runtime = float("inf")
    best_list = []

    iter_names = [
        d for d in os.listdir(iter_output_dir)
        if os.path.isdir(os.path.join(iter_output_dir, d)) and d.startswith("iter_")
    ]
    iter_names.sort(key=lambda x: numeric_suffix(x, "iter_"))

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
            metrics_file = os.path.join(attempts_dir, attempt_name, "metrics_artifacts.json")

            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)
                    runtime = metrics_data.get("metrics", {}).get("runtime")
                    if runtime is not None and runtime < best_runtime:
                        best_runtime = runtime
                except Exception:
                    pass

            best_list.append(None if best_runtime == float("inf") else best_runtime)

            if cumulative >= target_attempt:
                print(f"{task_path}, Runtime: {best_runtime}")
                return best_list

    # pad if fewer attempts than target
    while len(best_list) < target_attempt:
        best_list.append(None if best_runtime == float("inf") else best_runtime)

    return best_list


if __name__ == "__main__":
    sol_dest_dir.mkdir(parents=True, exist_ok=True)

    task_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("task")
    ]
    task_names.sort(key=lambda x: numeric_suffix(x, "task"))

    all_speedups = []  # list of lists: per-task speedup progress

    for task_name in task_names:
        task_number = numeric_suffix(task_name, "task")
        task_path = os.path.join(root_dir, task_name)
        eager = get_runtime(eager_runtimes, task_number)

        if eager is None:
            continue

        best_runtimes = best_runtimes_progress(task_path, target_attempt)
        speedup_progress = []
        for r in best_runtimes:
            if r is None or r <= 0:
                speedup_progress.append(1.0)  # neutral if missing
            else:
                s = eager / r
                # if task_number == 13:  # special case in your original
                #     s = 1.0
                if s < 1.0:
                    s = 1.0
                speedup_progress.append(s)

        all_speedups.append(speedup_progress)

    # transpose: list of [attempt_index][tasks]
    geomean_curve = []
    for attempt_idx in range(target_attempt):
        vals = [task_s[attempt_idx] for task_s in all_speedups if task_s[attempt_idx] is not None]
        if vals:
            geomean_curve.append(gmean(vals))
        else:
            geomean_curve.append(1.0)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(range(1, target_attempt+1), geomean_curve, label="Geomean speedup")
    plt.xlabel("Attempt Number")
    plt.ylabel("Speedup over PyTorch Eager")
    plt.title("OpenEvolve Agents Speedup by Attempt (Level 3-metr, H100)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Convergence plot saved to {plot_path}")
