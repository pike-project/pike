import os
import json
import shutil
from pathlib import Path
from scipy.stats import gmean

# Root directory
root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"
target_attempt = 300  # change as needed

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
eager_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/eager.json").resolve()
sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/openevolve_pop_25_agents_300/best_solutions").resolve()


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


def best_runtime_until_attempt(task_path, target_attempt):
    """
    Walk attempts in sorted numeric order, track the best runtime seen so far.
    Stop at target_attempt and return the best runtime and corresponding code.py path.
    """
    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        return None, None

    cumulative = 0
    best_runtime = float("inf")
    best_code_path = None

    # collect and sort iterations
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

        # collect and sort attempts
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
                except Exception:
                    # ignore malformed json
                    pass

            if cumulative == target_attempt:
                return (best_runtime if best_runtime != float("inf") else None,
                        best_code_path)

    return (None if best_runtime == float("inf") else best_runtime,
            best_code_path)


if __name__ == "__main__":
    # ensure destination exists
    sol_dest_dir.mkdir(parents=True, exist_ok=True)

    # collect tasks sorted numerically
    task_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("task")
    ]
    task_names.sort(key=lambda x: numeric_suffix(x, "task"))

    speedups = []

    for task_name in task_names:
        task_number = numeric_suffix(task_name, "task")  # int
        task_path = os.path.join(root_dir, task_name)
        best, best_code_path = best_runtime_until_attempt(task_path, target_attempt)
        eager = get_runtime(eager_runtimes, task_number)

        if best is not None and eager is not None and best > 0:
            speedup = eager / best
            speedups.append(speedup)
            print(f"{task_name} (id={task_number}): eager={eager}, best={best}, speedup={speedup:.3f}")

            # save best code
            if best_code_path:
                dest_file = sol_dest_dir / f"task_{task_number}.py"
                shutil.copy(best_code_path, dest_file)

        else:
            print(f"{task_name} (id={task_number}): missing data")

    if speedups:
        geo_mean = gmean(speedups)
        print(f"\nGeometric mean speedup across {len(speedups)} tasks = {geo_mean:.3f}")
    else:
        print("No valid speedups computed.")
