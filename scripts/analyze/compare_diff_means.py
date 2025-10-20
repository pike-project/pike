import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

diffs_dir = (curr_dir / "../../data/diffs").resolve()

figs_dir = (curr_dir / "../../results/ours/h100_level3-metr/results/figs").resolve()

run_name_1 = "h100_level_3-metr_prev_agents_trial_1"
run_name_2 = "h100_level_3-metr_openevolve_agents_trial_0"

label1 = "PIKE-B"
label2 = "PIKE-O"

# File paths
file1 = diffs_dir / run_name_1 / 'means.json'
file2 = diffs_dir / run_name_2 / 'means.json'

# --- Load JSON arrays ---
with open(file1, 'r') as f:
    array1 = json.load(f)

with open(file2, 'r') as f:
    array2 = json.load(f)

# --- Check if arrays are the same length ---
if len(array1) != len(array2):
    raise ValueError("Arrays must be the same length to compute elementwise difference.")

# --- Compute elementwise difference ---
diff = [a - b for a, b in zip(array1, array2)]
print(diff)

# --- Save the difference to JSON ---
# output_file = data_dir / 'difference.json'
# with open(output_file, 'w') as f:
#     json.dump(diff, f, indent=4)

# print(f"Elementwise difference saved to {output_file}")

# --- Compute mean of means ---
mean1 = np.mean(array1)
mean2 = np.mean(array2)

# --- Create histogram in the same theme ---
plt.figure(figsize=(4, 3))

plt.hist(
    array1,
    bins=20,
    alpha=0.5,
    label=label1,
    edgecolor="black"
)

plt.hist(
    array2,
    bins=20,
    alpha=0.5,
    label=label2,
    edgecolor="black"
)

# --- Add dashed vertical lines for mean of means ---
plt.axvline(mean1, color='blue', linestyle='--', linewidth=1)
plt.axvline(mean2, color='orange', linestyle='--', linewidth=1)

# --- Gridlines and labels ---
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel("Mean LoC Changed per Optimization Step")
plt.ylabel("Tasks")
plt.title("Mean Lines of Code (LoC) Changed")
plt.legend()
plt.tight_layout()

# --- Save the figure ---
hist_file = figs_dir / 'loc_hist.pdf'
plt.savefig(hist_file, format="pdf")
plt.close()

print(f"Histogram saved to {hist_file}")
