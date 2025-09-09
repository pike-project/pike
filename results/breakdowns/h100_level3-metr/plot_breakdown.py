import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
data_dir = (curr_dir / "data/runtimes").resolve()

# --- Load all runtimes ---
all_methods = {}
for file in data_dir.glob("*.json"):
    with open(file) as f:
        data = json.load(f)
    title = data["title"]
    results = {entry["problem_id"]: entry["runtime"] for entry in data["results"]}
    all_methods[title] = results

# Find eager baseline (case-insensitive)
eager_key = next((k for k in all_methods if k.lower() == "eager"), None)
if eager_key is None:
    raise ValueError("Missing baseline 'Eager.json' in data_dir")
eager_runtimes = all_methods[eager_key]

# --- Determine tasks from "ours (OpenEvolve)" ---
openevolve_key = next((k for k in all_methods if k.lower() == "ours (openevolve)"), None)
if openevolve_key is None:
    raise ValueError("Missing 'ours (OpenEvolve)' method in data_dir")
included_tasks = list(all_methods[openevolve_key].keys())

# --- Compute speedups ---
methods_speedups = {title: [] for title in all_methods if title != eager_key}

for task in included_tasks:
    eager_runtime = eager_runtimes.get(task)
    if eager_runtime is None:
        print(f"Warning: Task {task} missing baseline runtime, skipping.")
        continue

    for title, runtimes in all_methods.items():
        if title == eager_key:
            continue
        method_runtime = runtimes.get(task)
        if method_runtime is None or eager_runtime is None:
            speedup = 1.0
            print(f"Warning: Task {task}, method '{title}' has runtime None. Setting speedup=1.")
        else:
            speedup = eager_runtime / method_runtime
            if speedup < 1.0:
                speedup = 1.0
        methods_speedups[title].append(speedup)

# --- Task labels ---
task_labels_map = {}
level_dir = (curr_dir / "../../../KernelBench/level3-metr").resolve()
for filename in os.listdir(level_dir):
    if not filename.endswith(".py"):
        continue
    task = int(filename.split("_")[0])
    label = filename.split("_")[1].split(".py")[0]
    task_labels_map[task] = f"{task} ({label})"

labels = [task_labels_map.get(t, str(t)) for t in included_tasks]

# --- Sort tasks by "ours (OpenEvolve)" ascending ---
sort_indices = np.argsort(methods_speedups[openevolve_key])  # ascending
included_tasks = [included_tasks[i] for i in sort_indices]
labels_sorted = [labels[i] for i in sort_indices]
methods_speedups = {k: [v[i] for i in sort_indices] for k, v in methods_speedups.items()}

# --- Compute geometric mean ---
geomeans = {}
for name, values in methods_speedups.items():
    arr = np.array(values, dtype=float)
    arr = arr[arr > 0]  # filter out invalid/missing
    geomeans[name] = np.exp(np.mean(np.log(arr)))

# --- Plotting ---
x = np.arange(len(included_tasks))
fig, ax = plt.subplots(figsize=(12, 5.5))

for name, values in methods_speedups.items():
    ax.plot(
        x,
        values,
        label=f"{name} (gmean={geomeans[name]:.2f})",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
plt.title("Level 3-metr Speedup Over PyTorch Eager (A100)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
plt.ylabel("Speedup")
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.subplots_adjust(bottom=0.5)
ax.legend(loc='upper left', fontsize=8)

plt.yscale("log")

# --- Save plot ---
figs_dir = (curr_dir / "figs/breakdown").resolve()
os.makedirs(figs_dir, exist_ok=True)
fig.savefig(figs_dir / "individual_breakdown_new.pdf")

# --- Save CSV ---
df = pd.DataFrame(methods_speedups, index=labels_sorted)
df.reset_index(inplace=True)
df.rename(columns={"index": "Task"}, inplace=True)
csv_path = curr_dir / "data/speedups_table.csv"
df.to_csv(csv_path, index=False)

print("Geomean speedups:")
for k, v in geomeans.items():
    print(f"{k}: {v:.3f}")
print(f"CSV saved to: {csv_path}")
