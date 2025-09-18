import os

# Root directory
root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"

target_attempt = 300  # change as needed

def find_attempt_path(task_path, target_attempt):
    """
    Walk iter_* and attempt_* dirs in sorted order,
    return path to the nth attempt for this task.
    """
    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        return None

    cumulative = 0
    # sort iterations numerically by their suffix
    for iter_name in sorted(os.listdir(iter_output_dir),
                            key=lambda x: int(x.split("_")[1]) if x.startswith("iter_") else -1):
        iter_path = os.path.join(iter_output_dir, iter_name)
        if not os.path.isdir(iter_path) or not iter_name.startswith("iter_"):
            continue

        attempts_dir = os.path.join(iter_path, "attempts")
        if not os.path.exists(attempts_dir):
            continue

        # sort attempts numerically by their suffix
        attempt_names = sorted(
            [d for d in os.listdir(attempts_dir) if d.startswith("attempt_") and os.path.isdir(os.path.join(attempts_dir, d))],
            key=lambda x: int(x.split("_")[1])
        )

        for attempt_name in attempt_names:
            cumulative += 1
            if cumulative == target_attempt:
                return os.path.join(attempts_dir, attempt_name)

    return None


if __name__ == "__main__":
    for task_name in sorted(os.listdir(root_dir)):
        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path) or not task_name.startswith("task"):
            continue

        path = find_attempt_path(task_path, target_attempt)
        if path:
            print(f"{task_name}: {path}")
        else:
            print(f"{task_name}: target attempt {target_attempt} not found")
