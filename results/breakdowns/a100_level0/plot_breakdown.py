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

with open(data_dir / "comp_runtimes.json") as f:
    comp_runtimes = json.load(f)

with open(data_dir / "baseline_compile.json") as f:
    compile_runtimes = json.load(f)

with open(data_dir / "baseline_eager.json") as f:
    eager_runtimes = json.load(f)

with open(data_dir / "baseline_tensorrt.json") as f:
    tensorrt_runtimes = json.load(f)

# assert len(all_speedups_eager_final) == len(tasks_to_plot), "All speedups eager final length should be same as tasks_to_plot length"
# assert len(all_speedups_compile_final) == len(tasks_to_plot), "All speedups compile final length should be same as tasks_to_plot length"

# custom_speedups = [

# ]

# custom_labels = [

# ]

orig_speedups = [1.0, 1.1636128407775037, 15.474967591074936, 1.0006264385613004, 2.57470900944262]
openevolve_runtimes = [1.3629225250481067, 1.4859734816293408, 0.6134, 7.2187, 2.3093777851021495]

included_tasks = []
# v_rels = []
# compile_rels = []
# eager_rels = []

v_speedups = []
compile_speedups = []
tensorrt_speedups = []
our_orig_speedups = []
our_openevolve_speedups = []

tasks_to_plot = [1, 2, 3, 4, 5]

for idx, task in enumerate(tasks_to_plot):
    compile_runtime = get_baseline_runtime(compile_runtimes, task)
    eager_runtime = get_baseline_runtime(eager_runtimes, task)
    v_runtime = get_baseline_runtime(comp_runtimes, task)
    tensorrt_runtime = get_baseline_runtime(tensorrt_runtimes, task)

    if v_runtime is None:
        continue

    our_openevolve_speedup = eager_runtime / openevolve_runtimes[idx]
    v_speedup = eager_runtime / v_runtime

    v_speedups.append(eager_runtime / v_runtime)
    compile_speedups.append(eager_runtime / compile_runtime)
    tensorrt_speedups.append(eager_runtime / tensorrt_runtime)
    our_orig_speedups.append(orig_speedups[idx])
    our_openevolve_speedups.append(eager_runtime / openevolve_runtimes[idx])

    included_tasks.append(task)

# Pack all methods into a dict for convenience
methods = {
    "ours (OpenEvolve)": our_openevolve_speedups,
    "ours (orig)": our_orig_speedups,
    "Stanford blog": v_speedups,
    "torch.compile": compile_speedups,
    "TensorRT": tensorrt_speedups,
}

# --- Determine the "winner" method for each task ---
winners = []
for i, task in enumerate(included_tasks):
    values = {name: arr[i] for name, arr in methods.items()}
    winner = max(values, key=values.get)
    winners.append((task, winner, values))

# --- Sort tasks so that "ours (openevolve)" winners come first ---
winners_sorted = sorted(
    winners,
    key=lambda x: 0 if "openevolve" in x[1].lower() else 1
)

# --- Task label map ---
task_labels_map = {
    1: "1 (Conv)",
    2: "2 (Conv-ReLU-Pool)",
    3: "3 (LayerNorm)",
    4: "4 (MatMul)",
    5: "5 (Softmax)",
}

# After sorting tasks
tasks_sorted = [t for t, _, _ in winners_sorted]

# Use descriptive labels instead of numbers
labels_sorted = [task_labels_map[t] for t in tasks_sorted]

# Reorder all arrays consistently
def reorder(arr):
    return [arr[included_tasks.index(t)] for t in tasks_sorted]

# Reorder all method arrays consistently
methods_sorted = {name: reorder(arr) for name, arr in methods.items()}

# --- Plotting (dots only, offset horizontally) ---
x = np.arange(len(tasks_sorted))
offset = 0.15  # increased spacing since we now have 5 methods

fig, ax = plt.subplots(figsize=(7, 3.5))

marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', 'X'])

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

plt.title("Level 0 Runtimes Relative to PyTorch Eager (A100)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

plt.ylabel("Relative Runtime")
plt.yscale('log')
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

plt.subplots_adjust(bottom=0.25)
ax.legend(loc='upper right')

# --- Save ---
filename = "individual_breakdown_dots_sorted.pdf"
figs_dir = (curr_dir / "../../figs/breakdown").resolve()
os.makedirs(figs_dir, exist_ok=True)
save_path1 = figs_dir / filename
fig.savefig(save_path1)
