import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

diffs_dir = (curr_dir / "../../data/diffs").resolve()
figs_dir = (curr_dir / "../../results/ours/h100_level3-metr/results/figs").resolve()

# Ensure the output directory exists
os.makedirs(figs_dir, exist_ok=True)

run_name_1 = "h100_level_3-metr_prev_agents_trial_1"
run_name_2 = "h100_level_3-metr_openevolve_agents_trial_0"
# run_name_2 = "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands"

label1 = "PIKE-B"
label2 = "PIKE-O"

if run_name_2 == "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands":
    label2 = "PIKE-O (mut, nopar, noisl)"

# File paths
file1 = diffs_dir / run_name_1 / 'means.json'
file2 = diffs_dir / run_name_2 / 'means.json'

# --- Load JSON arrays ---
with open(file1, 'r') as f:
    array1 = json.load(f)

with open(file2, 'r') as f:
    array2 = json.load(f)

print(f"Array lens: {len(array1)}, {len(array2)}")

# --- Check if arrays are the same length ---
if len(array1) != len(array2):
    raise ValueError("Arrays must be the same length to compute elementwise difference.")

# --- Compute elementwise difference ---
diff = [a - b for a, b in zip(array1, array2)]
print(diff)

# --- Compute mean of means ---
mean1 = np.mean(array1)
mean2 = np.mean(array2)

# --- Create histogram in the same theme ---
if run_name_2 == "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands":
    plt.figure(figsize=(4, 2.25))
else:
    plt.figure(figsize=(4, 2))

# --- NEW: Compute a shared bin range for equal-width bins ---
# 1. Combine all data to find the global min and max
all_data = np.concatenate([array1, array2])
# 2. Create 20 equally spaced bins between the global min and max
#    We need 21 edges to create 20 bins.
# bin_edges = np.linspace(all_data.min(), all_data.max(), 21)
xlim_min = 50
xlim_max = 330
bin_edges = np.linspace(xlim_min, xlim_max, 26)

plt.grid(axis='y', linestyle='--', alpha=0.5)

# --- ADJUSTED: Use the shared bin_edges for both histograms ---
plt.hist(
    array1,
    bins=bin_edges,  # Use shared bin edges
    alpha=0.5,
    label=label1,
    edgecolor="black"
)

plt.hist(
    array2,
    bins=bin_edges,  # Use shared bin edges
    alpha=0.5,
    label=label2,
    edgecolor="black"
)

# --- Add dashed vertical lines for mean of means ---
plt.axvline(mean1, color='blue', linestyle='--', linewidth=1)
plt.axvline(mean2, color='orange', linestyle='--', linewidth=1)

# --- Gridlines and labels ---
plt.xlabel("Mean LoC Changed per Optimization Step")
plt.ylabel("Tasks")
# plt.title("Mean Lines of Code (LoC) Changed")

if run_name_2 == "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands":
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.46, 1.02),
        ncol=2,
        frameon=True
    )
else:
    plt.legend()

plt.tight_layout()

# plt.subplots_adjust(top=0.75)

# if run_name_2 == "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands":
#     plt.subplots_adjust(top=0.75)

plt.xlim(xlim_min, xlim_max)
plt.ylim(0, 9)

plt.yticks([0, 2, 4, 6, 8])

# --- Save the figure ---

hist_file = figs_dir / 'loc_hist.pdf'

if run_name_2 == "h100_level_3-metr_openevolve_agents_no_parallel_eval_no_islands":
    hist_file = figs_dir / 'loc_hist_mut.pdf'

plt.savefig(hist_file, format="pdf")
plt.close()

print(f"Histogram saved to {hist_file}")
