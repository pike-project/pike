import os
from collections import defaultdict
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# Root directory
# root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"
root_dir = (curr_dir / "../../../openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks").resolve()
# root_dir = (curr_dir / "../../data/parallel_runs/").resolve()

iter_attempt_counts = defaultdict(list)

total_attempts = 0
total_iters = 0

for task_name in os.listdir(root_dir):
    task_path = os.path.join(root_dir, task_name)
    if not os.path.isdir(task_path) or not task_name.startswith("task"):
        continue

    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        continue

    for iter_name in os.listdir(iter_output_dir):
        iter_path = os.path.join(iter_output_dir, iter_name)
        if not os.path.isdir(iter_path) or not iter_name.startswith("iter_"):
            continue

        attempts_dir = os.path.join(iter_path, "attempts")
        if not os.path.exists(attempts_dir):
            continue

        attempt_count = sum(
            os.path.isdir(os.path.join(attempts_dir, d)) and d.startswith("attempt_")
            for d in os.listdir(attempts_dir)
        )

        iter_attempt_counts[task_name].append(attempt_count)

# Per-task totals and averages
per_task_stats = {}
task_total_attempts_list = []  # store total attempts per task for mean across tasks
for task, counts in iter_attempt_counts.items():
    task_total_attempts = sum(counts)
    task_total_attempts_list.append(task_total_attempts)

    if counts:
        avg = sum(counts) / len(counts)
        pct_6 = sum(c == 6 for c in counts) / len(counts) * 100
        pct_gt1 = sum(c > 1 for c in counts) / len(counts) * 100
    else:
        avg, pct_6, pct_gt1 = 0, 0, 0
    per_task_stats[task] = (avg, pct_6, pct_gt1, task_total_attempts)

# Mean total attempts across all tasks
mean_attempts_across_tasks = sum(task_total_attempts_list) / len(task_total_attempts_list) if task_total_attempts_list else 0

# Sort by task number
sorted_tasks = sorted(per_task_stats.items(), key=lambda x: int(x[0].split("task")[1]))

print(f"Mean total attempts across all tasks: {mean_attempts_across_tasks:.2f}")
print("\nPer-task stats (sorted):")
for task, (avg, pct_6, pct_gt1, total_attempts_task) in sorted_tasks:
    print(f"  {task}: avg={avg:.2f}, % with 6 attempts={pct_6:.2f}%, % requiring >1 attempt={pct_gt1:.2f}%, total_attempts={total_attempts_task}")
