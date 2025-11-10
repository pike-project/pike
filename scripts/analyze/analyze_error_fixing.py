import os
import json
from collections import defaultdict
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

target_attempt = 300

# run_name = "h100_level_3-metr_prev_agents_trial_1"
# output_label = "prev_agents"

run_name = "h100_level_3-metr_prev_agents_cheap_efa_0"
output_label = "prev_agents_cheap_efa"

# run_name = "h100_level_3-metr_openevolve_agents_trial_0"
# output_label = "openevolve_agents"

# run_name = "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands"
# output_label = "openevolve_agents_no_parallel_eval_no_islands"

# run_name = "h100_level_3-metr_openevolve_agents_mutation_0"
# output_label = "openevolve_agents_mutation"
# run_name = "h100_level_3-metr_prev_agents_no_iba_0"
# output_label = "prev_agents_no_iba"

target_level = "3-metr"

task_blacklist_map = {
    "5": set(),
    "3-metr": {
        36,
        37,
        38,
        39,
        40,
        41,
        42,
    },
}

task_blacklist = task_blacklist_map.get(target_level, set())

target_dirname = f"h100_level{target_level}"

iter_attempt_data = defaultdict(list)

# total_attempts = 0
# total_iters = 0

error_fix_attempts_dir = (curr_dir / "../../results/ours/h100_level3-metr/results/data/error_fix_attempts").resolve()

os.makedirs(error_fix_attempts_dir, exist_ok=True)

# root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"
# root_dir = (curr_dir / "../../../openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks").resolve()
root_dir = (curr_dir / "../../data/parallel_runs" / run_name / "runs/runs/run_0/run/tasks").resolve()

for task_name in os.listdir(root_dir):
    if not task_name.startswith("task"):
        continue
    task = int(task_name.split("task")[1])

    if task in task_blacklist:
        continue

    task_path = os.path.join(root_dir, task_name)
    if not os.path.isdir(task_path):
        continue

    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        continue

    iter_names = os.listdir(iter_output_dir)
    # Sort the iteration directories numerically to process them in order
    iter_names_sorted = sorted(
        [d for d in iter_names if d.startswith("iter_")],
        key=lambda x: int(x.split("_")[1])
    )

    for iter_name in iter_names_sorted:
        iter_path = os.path.join(iter_output_dir, iter_name)
        if not os.path.isdir(iter_path):
            continue

        attempts_dir = Path(os.path.join(iter_path, "attempts"))
        if not os.path.exists(attempts_dir):
            continue

        attempt_dirnames = sorted(os.listdir(attempts_dir), key=lambda x: int(x.split("_")[1]))

        last_attempt_dirname = attempt_dirnames[-1]

        last_attempt_dir = attempts_dir / last_attempt_dirname
        error_fixing_success = False
        try:
            with open(last_attempt_dir / "metrics_artifacts.json") as f:
                metrics = json.load(f)
            
            if "metrics" in metrics and "runtime" in metrics["metrics"]:
                error_fixing_success = True
        except Exception:
            pass

        attempt_count = len(attempt_dirnames)

        iter_attempt_data[task].append((attempt_count, error_fixing_success))

total_count = 0
success_count = 0

for task, data in iter_attempt_data.items():
    counts_limited = []

    task_attempts = 0
    for (c, success) in data:
        total_count += 1
        if success:
            success_count += 1

        task_attempts += c
    
        if task_attempts > target_attempt:
            break

        counts_limited.append((c, success))
    
    data.clear()
    data += counts_limited

print(f"Success count: {success_count}, total count: {total_count}, Ratio: {success_count/total_count}")

with open(error_fix_attempts_dir / f"{output_label}.json", "w") as f:
    json.dump(iter_attempt_data, f)

# Per-task totals and averages
per_task_stats = {}
task_total_attempts_list = []  # store total attempts per task for mean across tasks
for task, data in iter_attempt_data.items():
    counts = []
    for (c, _) in data:
        counts.append(c)

    task_total_attempts = sum(counts)
    task_total_attempts_list.append(task_total_attempts)

    if counts:
        avg = sum(counts) / len(counts)
        pct_6 = sum(c == 6 for c in counts) / len(counts) * 100
        pct_gt1 = sum(c > 1 for c in counts) / len(counts) * 100
    else:
        avg, pct_6, pct_gt1 = 0, 0, 0
    per_task_stats[task] = (avg, pct_6, pct_gt1, task_total_attempts)

# Sort by task number
sorted_tasks = sorted(per_task_stats.items(), key=lambda x: x[0])

print("\nPer-task stats (sorted):")
for task, (avg, pct_6, pct_gt1, total_attempts_task) in sorted_tasks:
    print(f"  {task}: avg={avg:.2f}, % with 6 attempts={pct_6:.2f}%, % requiring >1 attempt={pct_gt1:.2f}%, total_attempts={total_attempts_task}")
