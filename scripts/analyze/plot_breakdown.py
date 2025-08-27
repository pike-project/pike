import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import argparse
import math
from scipy.stats import gmean

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_dir = (curr_dir / "../../data/runs/final_run2").resolve()

def get_baseline_runtime(data, task):
    for v in data:
        if v["problem_id"] == task:
            try:
                return v["results"]["eval_results"]["runtime"]
            except Exception as e:
                return None
    
    return None

with open(run_dir / "comp_runtimes.json") as f:
    comp_runtimes = json.load(f)

with open(run_dir / "baseline_compile.json") as f:
    compile_runtimes = json.load(f)

with open(run_dir / "baseline_eager.json") as f:
    eager_runtimes = json.load(f)

# assert len(all_speedups_eager_final) == len(tasks_to_plot), "All speedups eager final length should be same as tasks_to_plot length"
# assert len(all_speedups_compile_final) == len(tasks_to_plot), "All speedups compile final length should be same as tasks_to_plot length"

orig_speedups = [1.0, 1.1636128407775037, 15.474967591074936, 1.0006264385613004, 2.57470900944262]
openevolve_runtimes = [1.3629225250481067, 1.4859734816293408, 0.6134, 7.2187, 2.3093777851021495]

included_tasks = []
# v_rels = []
# compile_rels = []
# eager_rels = []

v_speedups = []
compile_speedups = []
our_orig_speedups = []
our_openevolve_speedups = []

tasks_to_plot = [1, 2, 3, 4, 5]

for idx, task in enumerate(tasks_to_plot):
    compile_runtime = get_baseline_runtime(compile_runtimes, task)
    eager_runtime = get_baseline_runtime(eager_runtimes, task)
    v_runtime = get_baseline_runtime(comp_runtimes, task)

    if v_runtime is None:
        continue
    
    # our_orig_speedup = orig_speedups_eager[idx]
    our_orig_speedup = orig_speedups[idx]
    our_openevolve_speedup = eager_runtime / openevolve_runtimes[idx]
    v_speedup = eager_runtime / v_runtime
    compile_speedup = eager_runtime / compile_runtime

    # v_rel = our_speedup / v_speedup
    # compile_rel = our_speedup / compile_speedup

    # v_rels.append(v_rel)
    # compile_rels.append(compile_rel)
    # eager_rels.append(our_speedup)
    v_speedups.append(v_speedup)
    compile_speedups.append(compile_speedup)
    our_orig_speedups.append(our_orig_speedup)
    our_openevolve_speedups.append(our_openevolve_speedup)

    included_tasks.append(task)

fig, ax = plt.subplots(figsize=(6, 2.5))

ax.plot(included_tasks, v_speedups, label="Stanford blog", marker='o', markersize=4)
ax.plot(included_tasks, compile_speedups, label="torch.compile", marker='o', markersize=4)
ax.plot(included_tasks, our_orig_speedups, label="ours (orig)", marker='o', markersize=4)
ax.plot(included_tasks, our_openevolve_speedups, label="ours (openevolve)", marker='o', markersize=4)

plt.title("Level 0 Runtimes Relative to PyTorch Eager (A100)")

plt.xticks(range(1, 6))
plt.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5, alpha=0.3)
plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.3)

plt.ylabel("Task Number")
plt.ylabel("Relative Runtime")

plt.yscale('log')
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

ax.legend(loc='upper right')

filename = "individual_breakdown.pdf"

figs_dir = (curr_dir / "../../figs/breakdown").resolve()

os.makedirs(figs_dir, exist_ok=True)
save_path1 = figs_dir / filename
fig.savefig(save_path1)
