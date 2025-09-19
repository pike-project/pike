import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# run_dir = Path("/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_runs/level3-metr_trial0")
# run_dir = Path("/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_1")
run_dir = Path("/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_4/tasks")

eager_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/eager.json").resolve()

with open(eager_path) as f:
    eager_runtimes = json.load(f)

def get_runtime(data, task):
    for v in data["results"]:
        if v["problem_id"] == task:
            return v["runtime"]
    
    return None

tasks = []
for dirname in os.listdir(run_dir):
    tasks.append(int(dirname.split("task")[1]))

tasks = sorted(tasks)

all_checkpoints = []
speedups = []

for task in tasks:
    checkpoints_dir = run_dir / f"task{task}/output/checkpoints"

    eager_runtime = get_runtime(eager_runtimes, task)

    if eager_runtime is None:
        raise Exception(f"Task {task} eager runtime is None")

    checkpoints = []
    for dirname in os.listdir(checkpoints_dir):
        checkpoints.append(int(dirname.split("_")[1]))

    checkpoints = sorted(checkpoints)

    task_speedups = []

    for checkpoint in checkpoints:
        best_info_path = checkpoints_dir / f"checkpoint_{checkpoint}/best_program_info.json"

        with open(best_info_path) as f:
            best_info = json.load(f)

        runtime = best_info["metrics"]["runtime"]

        speedup = eager_runtime / runtime

        task_speedups.append(speedup)

        # print(f"Task: {task}, Runtime: {runtime}")

    if len(task_speedups) == 0:
        raise Exception(f"Task {task} has no checkpoint speedups")

    speedups.append(task_speedups)
    all_checkpoints.append(checkpoints)

max_checkpoints = None
max_speedups_count = 0
for (idx, t) in enumerate(speedups):
    if len(t) > max_speedups_count:
        max_speedups_count = len(t)
        max_checkpoints = all_checkpoints[idx]

if max_checkpoints is None:
    raise Exception("max_checkpoints is None")

for t in speedups:
    while len(t) < max_speedups_count:
        t.append(t[-1])

speedups_np = np.array(speedups)

speedups_gmean = gmean(speedups_np)

plt.plot(max_checkpoints, speedups_gmean, linestyle='-', color="#ff6583")

# Add labels and title
plt.xlabel("Attempt Number")
plt.ylabel("Speedup Over PyTorch Eager")
plt.title("OpenEvolve Speedup By Attempt (Level 3-metr, H100)")

plt.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5, alpha=0.3)
plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.3)

filename = "convergence5.pdf"
figs_dir = (curr_dir / "../../results/breakdowns/h100_level3-metr/figs").resolve()
os.makedirs(figs_dir, exist_ok=True)
save_path = figs_dir / filename
plt.savefig(save_path)
