import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

data_dir = (curr_dir / "../../data/diff_means").resolve()

# File paths
file1 = data_dir / 'means1.json'
file2 = data_dir / 'means2.json'

# Labels for each file (customizable)
label_title_map = [
    ("means1.json", "PIKE-B"),
    ("means2.json", "PIKE-O"),
]

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
output_file = data_dir / 'difference.json'
with open(output_file, 'w') as f:
    json.dump(diff, f, indent=4)

print(f"Elementwise difference saved to {output_file}")

# --- Compute mean of means ---
mean1 = np.mean(array1)
mean2 = np.mean(array2)

# --- Create histogram in the same theme ---
plt.figure(figsize=(4, 3))

plt.hist(
    array1,
    bins=20,
    alpha=0.5,
    label=label_title_map[0][1],
    edgecolor="black"
)

plt.hist(
    array2,
    bins=20,
    alpha=0.5,
    label=label_title_map[1][1],
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
hist_file = data_dir / 'hist.pdf'
plt.savefig(hist_file, format="pdf")
plt.close()

print(f"Histogram saved to {hist_file}")
