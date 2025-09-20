import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
data_dir = (curr_dir / "data/runtimes").resolve()

included_files = ["eager", "oe_agents", "compile"]
# included_files = ["eager", "ours_openevolve", "orig"]
# included_files = ["eager", "ours_openevolve", "metr"]
# included_files = ["eager", "ours_openevolve", "compile", "tensorrt"]

primary_str_match = "ours (openevolve, agents)"
# primary_str_match = "ours (openevolve)"
# primary_str_match = "ours (prev. agent-based)"

# --- Load all runtimes ---
all_methods = {}
file_to_title_map = {}

for file in data_dir.glob("*.json"):
    if file.stem not in included_files:
        continue

    with open(file) as f:
        data = json.load(f)
    title = data["title"]
    results = {entry["problem_id"]: entry["runtime"] for entry in data["results"]}
    all_methods[title] = results

    # Build map dynamically
    file_to_title_map[file.stem] = title

# Find eager baseline
eager_title = file_to_title_map.get("eager", None)
if eager_title is None:
    raise ValueError("Missing baseline 'Eager.json' in data_dir")
eager_key = next((k for k in all_methods if k.lower() == eager_title.lower()), None)
eager_runtimes = all_methods[eager_key]

# --- Determine tasks from our primary sorting key ---
primary_key = next((k for k in all_methods if k.lower() == primary_str_match), None)
if primary_key is None:
    raise ValueError("Missing primary key method in data_dir")
included_tasks = list(all_methods[primary_key].keys())

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
    # task_labels_map[task] = f"{task} ({label})"
    task_labels_map[task] = label

labels = [task_labels_map.get(t, str(t)) for t in included_tasks]

# --- Sort tasks by primary key ascending ---
sort_indices = np.argsort(methods_speedups[primary_key])  # ascending
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
plot_mode = "bar"  # choose "line" or "bar"
x = np.arange(len(included_tasks))
fig, ax = plt.subplots(figsize=(12, 5.5))

# Enforce plotting order using included_files → titles
plot_order = []
for f in included_files:
    if f == "eager":  # skip baseline
        continue
    title = file_to_title_map.get(f, None)
    if title and title in methods_speedups:
        plot_order.append(title)

if plot_mode == "line":
    for name in plot_order:
        values = methods_speedups[name]
        ax.plot(
            x,
            values,
            label=f"{name} (gmean={geomeans[name]:.2f})",
            linewidth=2,
        )
elif plot_mode == "bar":
    n_methods = len(plot_order)
    width = 0.8 / n_methods  # total width of the group is 0.8

    for i, name in enumerate(plot_order):
        values = methods_speedups[name]
        offsets = x - 0.4 + i * width + width / 2  # center the group around x
        ax.bar(
            offsets,
            values,
            width=width,
            label=f"{name} (gmean={geomeans[name]:.2f})",
            alpha=0.9
        )
else:
    raise ValueError(f"Unknown plot_mode: {plot_mode}")

ax.set_xticks(x)
ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
plt.title("Level 3-metr Speedup Over PyTorch Eager (H100)")
plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
plt.ylabel("Speedup")

plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.95)
ax.legend(loc='upper left', fontsize=10)
plt.yscale("log")

# --- Save plot ---
figs_dir = (curr_dir / "figs/breakdown").resolve()
os.makedirs(figs_dir, exist_ok=True)
fig.savefig(figs_dir / f"individual_breakdown_{plot_mode}.pdf")

# --- Save CSV ---
# Enforce CSV columns order the same as included_files
ordered_cols = [file_to_title_map[f] for f in included_files if f != "eager" and f in file_to_title_map]
df = pd.DataFrame({name: methods_speedups[name] for name in ordered_cols}, index=labels_sorted)
df.reset_index(inplace=True)
df.rename(columns={"index": "Task"}, inplace=True)

# Append geomean row
# geo_row = {"Task": "Geomean"}
# for k in ordered_cols:
#     geo_row[k] = geomeans[k]
# df = pd.concat([df, pd.DataFrame([geo_row])], ignore_index=True)

csv_path = curr_dir / "data/speedups_table_2.csv"
df.to_csv(csv_path, index=False)

print("Geomean speedups:")
for k in ordered_cols:
    print(f"{k}: {geomeans[k]:.3f}")
print(f"CSV saved to: {csv_path}")
