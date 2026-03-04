import os
import json
# difflib, shutil, subprocess, numpy, radon are not used in this snippet
# but are kept in case they are used elsewhere in your script.
import difflib
import shutil
import subprocess
from pathlib import Path
import numpy as np
from radon.raw import analyze
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_name = "h100_level_3-metr_prev_agents_trial_1"
output_data_path = (curr_dir / f"results/data/sloc_speedup/{run_name}.json").resolve()

# Create a dummy data file for demonstration if it doesn't exist
if not output_data_path.exists():
    print(f"Dummy data file created at: {output_data_path}")
    os.makedirs(output_data_path.parent, exist_ok=True)
    dummy_data = [
        {"task": i, "slocs": np.random.randint(10, 100, 15).tolist(), "speedups": (np.random.rand(15) * 5 + 1 + np.arange(15)*0.1).tolist()}
        for i in range(28) # Create data for 28 plots
    ]
    # Added a slight trend to the dummy data to make the linear fit more visible
    with open(output_data_path, 'w') as f:
        json.dump(dummy_data, f)


with open(output_data_path) as f:
    data = json.load(f)

all_slocs = []
all_speedups = []

# 4. Loop through the data and the axes simultaneously
for i, v in enumerate(data):
    task = v["task"]
    slocs = np.array(v["slocs"])
    speedups = np.array(v["speedups"])

    if len(slocs) < 2 or len(speedups) < 2:
        continue

    # Normalize SLOCs to the range [0, 1]
    min_sloc, max_sloc = slocs.min(), slocs.max()
    if max_sloc == min_sloc:
        # If all values are the same, set them to a mid-point (0.5) to avoid division by zero
        normalized_slocs = np.full_like(slocs, 0.5, dtype=float)
    else:
        normalized_slocs = (slocs - min_sloc) / (max_sloc - min_sloc)

    # Normalize speedups to the range [0, 1]
    min_speedup, max_speedup = speedups.min(), speedups.max()
    if max_speedup == min_speedup:
        normalized_speedups = np.full_like(speedups, 0.5, dtype=float)
    else:
        normalized_speedups = (speedups - min_speedup) / (max_speedup - min_speedup)

    all_slocs += normalized_slocs.tolist()
    all_speedups += normalized_speedups.tolist()

fig, ax = plt.subplots(figsize=(4, 3.25))

bins = 20
heatmap = ax.hist2d(all_slocs, all_speedups, bins=bins, cmap='viridis') # , norm=LogNorm()

cbar = fig.colorbar(heatmap[3], ax=ax)
cbar.set_label('Density of Data Points (Count)', fontsize=12)

# plt.scatter(all_slocs, all_speedups)

plt.xlabel("Normalized SLOC")
plt.ylabel("Normalized Speedup")
# plt.grid(True, linestyle='--', alpha=0.6)

# Set axis limits to be explicitly [0, 1] for clarity
plt.xlim(0, 1)
plt.ylim(0, 1)

# 7. Define the output directory for the combined figure
#    Saving to a different directory to avoid confusion with the single task plots
figs_dir = curr_dir / "results/figs/sloc_speedup_scatter"
os.makedirs(figs_dir, exist_ok=True)

# 8. Adjust layout to prevent titles and labels from overlapping
plt.tight_layout()

# 9. Save the entire figure to a single file, outside the loop
output_fig_path = figs_dir / f"{run_name}_normalized_joined.pdf"
plt.savefig(output_fig_path)

print(f"Normalized subplots figure saved to: {output_fig_path}")

# To display the plot in an interactive session (e.g., Jupyter)
# plt.show()
