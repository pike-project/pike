import os
from collections import defaultdict
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

USE_OPENEVOLVE_STRUCTURE = False
target_attempt = 300

iter_attempt_counts = defaultdict(list)

# total_attempts = 0
# total_iters = 0

if USE_OPENEVOLVE_STRUCTURE:
    # root_dir = "/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks"
    root_dir = (curr_dir / "../../../openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks").resolve()

    for task_name in os.listdir(root_dir):
        if not task_name.startswith("task"):
            continue
        task = int(task_name.split("task")[1])

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

            attempts_dir = os.path.join(iter_path, "attempts")
            if not os.path.exists(attempts_dir):
                continue

            attempt_count = sum(
                os.path.isdir(os.path.join(attempts_dir, d)) and d.startswith("attempt_")
                for d in os.listdir(attempts_dir)
            )

            iter_attempt_counts[task].append(attempt_count)
else:
    root_dir = (curr_dir / "../../data/parallel_runs/h100_level_3-metr_prev_noagents_trial_1/runs/runs/run_0/run/levels/level_3-metr").resolve()

    for task_name in os.listdir(root_dir):
        if not task_name.startswith("task"):
            continue
        task = int(task_name.split("_")[1])

        task_path = os.path.join(root_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        phases_dir = os.path.join(task_path, "phases")
        if not os.path.exists(phases_dir):
            continue

        phase_names = os.listdir(phases_dir)
        phase_numbers = sorted([
            int(p.split("_")[1]) for p in phase_names if p.startswith("phase_")
        ])

        for phase_number in phase_numbers:
            phase_path = os.path.join(phases_dir, f"phase_{phase_number}")
            if not os.path.isdir(phase_path):
                continue

            agents_dir = Path(phase_path) / "agents"
            if not os.path.exists(agents_dir):
                continue

            agent_names = sorted(
                [d for d in os.listdir(agents_dir) if d.startswith("agent_")],
                key=lambda x: int(x.split("_")[1])
            )

            # TODO: finish iterating here, where the attempt count is the number of steps for a given agent
            for agent_name in agent_names:
                agent_path = agents_dir / agent_name

                if not os.path.isdir(agent_path):
                    continue

                # The number of steps for a given agent is the attempt count for that "run"
                step_count = sum(
                    os.path.isdir(agent_path / step) and step.startswith("step_")
                    for step in os.listdir(agent_path)
                )

                iter_attempt_counts[task].append(step_count)


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
sorted_tasks = sorted(per_task_stats.items(), key=lambda x: x[0])

print(f"Mean total attempts across all tasks: {mean_attempts_across_tasks:.2f}")
print("\nPer-task stats (sorted):")
for task, (avg, pct_6, pct_gt1, total_attempts_task) in sorted_tasks:
    print(f"  {task}: avg={avg:.2f}, % with 6 attempts={pct_6:.2f}%, % requiring >1 attempt={pct_gt1:.2f}%, total_attempts={total_attempts_task}")
