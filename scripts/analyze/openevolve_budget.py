import os
import json

# Root directory
root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"

target_attempt = 300  # change as needed

def numeric_suffix(name: str, prefix: str) -> int:
    """Extract integer suffix from names like 'task12', 'iter_34', 'attempt_5'."""
    try:
        return int(name.replace(prefix, ""))
    except ValueError:
        return -1  # for safety

def best_runtime_until_attempt(task_path, target_attempt):
    """
    Walk attempts in sorted numeric order, track the best runtime seen so far.
    Stop at target_attempt and return the best runtime.
    """
    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        return None

    cumulative = 0
    best_runtime = float("inf")

    # collect iterations
    iter_names = []
    for d in os.listdir(iter_output_dir):
        if os.path.isdir(os.path.join(iter_output_dir, d)) and d.startswith("iter_"):
            iter_names.append(d)

    # sort numerically by iter number
    iter_names.sort(key=lambda x: numeric_suffix(x, "iter_"))

    for iter_name in iter_names:
        iter_path = os.path.join(iter_output_dir, iter_name)
        attempts_dir = os.path.join(iter_path, "attempts")
        if not os.path.exists(attempts_dir):
            continue

        # collect attempts
        attempt_names = []
        for d in os.listdir(attempts_dir):
            if os.path.isdir(os.path.join(attempts_dir, d)) and d.startswith("attempt_"):
                attempt_names.append(d)

        # sort numerically by attempt number
        attempt_names.sort(key=lambda x: numeric_suffix(x, "attempt_"))

        for attempt_name in attempt_names:
            cumulative += 1
            attempt_path = os.path.join(attempts_dir, attempt_name)
            metrics_file = os.path.join(attempt_path, "metrics_artifacts.json")

            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        metrics_data = json.load(f)
                    runtime = metrics_data.get("metrics", {}).get("runtime")
                    if runtime is not None and runtime < best_runtime:
                        best_runtime = runtime
                except Exception:
                    # ignore malformed json, keep current best_runtime
                    pass

            if cumulative == target_attempt:
                return best_runtime if best_runtime != float("inf") else None

    return None


if __name__ == "__main__":
    # collect tasks
    task_names = []
    for d in os.listdir(root_dir):
        task_path = os.path.join(root_dir, d)
        if os.path.isdir(task_path) and d.startswith("task"):
            task_names.append(d)

    # sort numerically by task number
    task_names.sort(key=lambda x: numeric_suffix(x, "task"))

    results = {}
    for task_name in task_names:
        task_path = os.path.join(root_dir, task_name)
        results[task_name] = best_runtime_until_attempt(task_path, target_attempt)

    # print in numeric task order
    for task_name in task_names:
        best = results[task_name]
        if best is not None:
            print(f"{task_name}: best runtime until attempt {target_attempt} = {best}")
        else:
            print(f"{task_name}: target attempt {target_attempt} not found")
