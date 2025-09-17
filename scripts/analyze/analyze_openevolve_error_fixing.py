import os
from collections import defaultdict

# Updated root directory
root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_3/tasks"

iter_attempt_counts = defaultdict(list)

total_attempts = 0
total_iters = 0
iters_with_6_attempts = 0
iters_with_gt1_attempt = 0

for task_name in os.listdir(root_dir):
    task_path = os.path.join(root_dir, task_name)
    if not os.path.isdir(task_path):
        continue
    if not task_name.startswith("task"):
        continue

    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        continue

    for iter_name in os.listdir(iter_output_dir):
        iter_path = os.path.join(iter_output_dir, iter_name)
        if not os.path.isdir(iter_path):
            continue
        if not iter_name.startswith("iter_"):
            continue

        attempts_dir = os.path.join(iter_path, "attempts")
        if not os.path.exists(attempts_dir):
            continue

        attempt_count = sum(
            os.path.isdir(os.path.join(attempts_dir, d)) and d.startswith("attempt_")
            for d in os.listdir(attempts_dir)
        )

        iter_attempt_counts[task_name].append(attempt_count)
        total_attempts += attempt_count
        total_iters += 1
        if attempt_count == 6:
            iters_with_6_attempts += 1
        if attempt_count > 1:
            iters_with_gt1_attempt += 1

# Global stats
global_avg = total_attempts / total_iters if total_iters > 0 else 0
global_pct_6 = (iters_with_6_attempts / total_iters * 100) if total_iters > 0 else 0
global_pct_gt1 = (iters_with_gt1_attempt / total_iters * 100) if total_iters > 0 else 0

# Per-task averages + percentages
per_task_stats = {}
for task, counts in iter_attempt_counts.items():
    if counts:
        avg = sum(counts) / len(counts)
        pct_6 = sum(c == 6 for c in counts) / len(counts) * 100
        pct_gt1 = sum(c > 1 for c in counts) / len(counts) * 100
    else:
        avg, pct_6, pct_gt1 = 0, 0, 0
    per_task_stats[task] = (avg, pct_6, pct_gt1)

# Sort by task number
sorted_tasks = sorted(per_task_stats.items(), key=lambda x: int(x[0].split("task")[1]))

print("Global average attempts per iter:", global_avg)
print(f"Global % of iters with 6 attempts: {global_pct_6:.2f}%")
print(f"Global % of iters requiring >1 attempt: {global_pct_gt1:.2f}%")
print("\nPer-task stats (sorted):")
for task, (avg, pct_6, pct_gt1) in sorted_tasks:
    print(f"  {task}: avg={avg:.2f}, % with 6 attempts={pct_6:.2f}%, % requiring >1 attempt={pct_gt1:.2f}%")
