import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import itertools
import subprocess
import argparse
import math
from scipy.stats import gmean

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

data_dir = (curr_dir / "data").resolve()

def get_baseline_runtime(data, task):
    for v in data:
        if v["problem_id"] == task:
            try:
                return v["results"]["eval_results"]["runtime"]
            except Exception as e:
                return None
    
    return None

with open(data_dir / "ours_openevolve.json") as f:
    comp_runtimes = json.load(f)

with open(data_dir / "compile.json") as f:
    compile_runtimes = json.load(f)

with open(data_dir / "eager.json") as f:
    eager_runtimes = json.load(f)

with open(data_dir / "tensorrt.json") as f:
    tensorrt_runtimes = json.load(f)

with open(data_dir / "metr.json") as f:
    metr_runtimes = json.load(f)

# assert len(all_speedups_eager_final) == len(tasks_to_plot), "All speedups eager final length should be same as tasks_to_plot length"
# assert len(all_speedups_compile_final) == len(tasks_to_plot), "All speedups compile final length should be same as tasks_to_plot length"

# custom_speedups = [

# ]

# custom_labels = [

# ]

included_tasks = []
# v_rels = []
# compile_rels = []
# eager_rels = []

v_speedups = []
compile_speedups = []
tensorrt_speedups = []
metr_speedups = []
our_orig_speedups = []
our_openevolve_speedups = []

tasks_to_plot = [1, 2, 4, 6, 8, 9, 10, 13, 14, 15, 16, 17, 19, 20, 21]

for idx, task in enumerate(tasks_to_plot):
    eager_runtime = get_baseline_runtime(eager_runtimes, task)
    compile_runtime = get_baseline_runtime(compile_runtimes, task)
    v_runtime = get_baseline_runtime(comp_runtimes, task)
    tensorrt_runtime = get_baseline_runtime(tensorrt_runtimes, task)
    metr_runtime = get_baseline_runtime(metr_runtimes, task)

    if metr_runtime is None:
        metr_runtime = eager_runtime

    if tensorrt_runtime is None:
        tensorrt_runtime = eager_runtime

    if eager_runtime is None:
        continue

    v_speedup = eager_runtime / v_runtime

    v_speedups.append(eager_runtime / v_runtime)
    compile_speedups.append(eager_runtime / compile_runtime)
    tensorrt_speedups.append(eager_runtime / tensorrt_runtime)
    metr_speedups.append(eager_runtime / metr_runtime)
    # our_orig_speedups.append(orig_speedups[idx])
    # our_openevolve_speedups.append(eager_runtime / openevolve_runtimes[idx])

    included_tasks.append(task)

# Pack all methods into a dict for convenience
methods = {
    "ours (OpenEvolve)": v_speedups,
    # "ours (orig)": our_orig_speedups,
    # "Stanford blog": v_speedups,
    "METR": metr_speedups,
    "torch.compile": compile_speedups,
    "TensorRT": tensorrt_speedups,
}

# --- Determine the "winner" method for each task ---
winners = []
for i, task in enumerate(included_tasks):
    values = {name: arr[i] for name, arr in methods.items()}
    winner = max(values, key=values.get)
    winners.append((task, winner, values))

# --- Sort tasks by "ours (OpenEvolve)" speedup descending ---
# winners_sorted = sorted(
#     winners,
#     key=lambda x: x[2]["ours (OpenEvolve)"],  # x[2] contains the values dict
#     reverse=True  # highest speedup first
# )
winners_sorted = winners

# --- Task label map ---
# task_labels_map = {
#     1: "1 (Conv)",
#     2: "2 (Conv-ReLU-Pool)",
#     3: "3 (LayerNorm)",
#     4: "4 (MatMul)",
#     5: "5 (Softmax)",
# }
task_labels_map = {}
level_dir = (curr_dir / "../../../KernelBench/level3-metr").resolve()

for filename in os.listdir(level_dir):
    if not filename.endswith(".py"):
        continue

    task = int(filename.split("_")[0])
    label = filename.split("_")[1].split(".py")[0]

    task_labels_map[task] = f"{task} ({label})"

# After sorting tasks
tasks_sorted = [t for t, _, _ in winners_sorted]

# Use descriptive labels instead of numbers
labels_sorted = [task_labels_map[t] for t in tasks_sorted]
# labels_sorted = [f"{t}" for t in tasks_sorted]

# Reorder all arrays consistently
def reorder(arr):
    return [arr[included_tasks.index(t)] for t in tasks_sorted]

# Reorder all method arrays consistently
methods_sorted = {name: reorder(arr) for name, arr in methods.items()}

# --- Plotting (dots only, offset horizontally) ---
x = np.arange(len(tasks_sorted))
offset = 0.15  # increased spacing since we now have 5 methods

fig, ax = plt.subplots(figsize=(10, 3.5))

marker_cycle = itertools.cycle(['o', 'D', '^', 'v', 'P', 'X'])

# Enumerate through methods and plot with offsets
for i, (name, values) in enumerate(methods_sorted.items()):
    ax.scatter(x + (i - (len(methods_sorted)-1)/2) * offset,
               values,
               label=name,
               marker=next(marker_cycle),
               s=30)

# Add vertical separators between task groups
for i in range(len(tasks_sorted) - 1):
    ax.axvline(x=i + 0.5, color='lightgray', linestyle='--', linewidth=0.8, alpha=0.6)

# --- Formatting ---
ax.set_xticks(x)
ax.set_xticklabels(labels_sorted, rotation=20, ha="right")

plt.title("Level 3-metr Speedup Over PyTorch Eager (A100)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

plt.ylabel("Speedup")
# plt.yscale('log')
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

plt.subplots_adjust(bottom=0.35)
ax.legend(loc='upper right')

# --- Save ---
filename = "individual_breakdown.pdf"
figs_dir = (curr_dir / "figs/breakdown").resolve()
os.makedirs(figs_dir, exist_ok=True)
save_path1 = figs_dir / filename
fig.savefig(save_path1)

# Create DataFrame
df = pd.DataFrame(methods_sorted, index=labels_sorted)

# Reset index to have "Task" as a column
df.reset_index(inplace=True)
df.rename(columns={"index": "Task"}, inplace=True)

# Export to CSV
csv_path = curr_dir / "speedups_table.csv"
df.to_csv(csv_path, index=False)

print(f"CSV saved to: {csv_path}")
