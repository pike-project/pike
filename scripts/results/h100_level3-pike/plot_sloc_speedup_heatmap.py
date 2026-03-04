import os
import json
import difflib
import shutil
import subprocess
from pathlib import Path
import numpy as np
from radon.raw import analyze
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # Import LogNorm for better color scaling

# Use a non-interactive backend for running on servers without a display
# plt.switch_backend('Agg') 

curr_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

run_name = "h100_level_3-metr_prev_agents_trial_1"
output_data_path = (curr_dir / f"results/data/sloc_speedup/{run_name}.json").resolve()

# Create a dummy data file for demonstration if it doesn't exist
if not output_data_path.exists():
    print(f"Dummy data file created at: {output_data_path}")
    os.makedirs(output_data_path.parent, exist_ok=True)
    dummy_data = [
        {"task": i, "slocs": np.random.randint(10, 100, 15).tolist(), "speedups": (np.random.rand(15) * 5 + 1 + np.arange(15)*0.1).tolist()}
        for i in range(28) # Create data for 28 tasks
    ]
    with open(output_data_path, 'w') as f:
        json.dump(dummy_data, f)


with open(output_data_path) as f:
    data = json.load(f)

# --- Data Aggregation and Normalization ---

# 1. Join all the data from individual tasks into two master lists
all_slocs = []
all_speedups = []
for task_data in data:
    all_slocs.extend(task_data["slocs"])
    all_speedups.extend(task_data["speedups"])

# Convert lists to numpy arrays for efficient processing
x_data = np.array(all_slocs)
y_data = np.array(all_speedups)

# 2. Normalize the entire combined dataset to a [0, 1] scale
x_normalized = (x_data - x_data.min()) / (x_data.max() - x_data.min())
y_normalized = (y_data - y_data.min()) / (y_data.max() - y_data.min())

# --- Heatmap Plotting ---

# 3. Create a figure and axis for the heatmap
fig, ax = plt.subplots(figsize=(12, 9))

# 4. Create the 2D histogram (heatmap)
#    - bins: The number of divisions on each axis. More bins = higher resolution.
#    - cmap: The colormap to use. 'viridis' or 'inferno' are good choices.
#    - norm=LogNorm(): This is crucial for heatmaps where some bins have vastly
#      more points than others. It makes lower-density areas more visible.
bins = 50
heatmap = ax.hist2d(x_normalized, y_normalized, bins=bins, cmap='viridis', norm=LogNorm())

# 5. Add a color bar to show the density scale
cbar = fig.colorbar(heatmap[3], ax=ax)
cbar.set_label('Density of Data Points (Count)', fontsize=12)

# 6. Configure the plot's appearance
ax.set_title(f"Heatmap of Normalized SLOC vs. Normalized Speedup (All Tasks)\nRun: {run_name}", fontsize=16)
ax.set_xlabel("Normalized SLOC (across all tasks)", fontsize=12)
ax.set_ylabel("Normalized Speedup (across all tasks)", fontsize=12)

# Set plot limits to the normalized range [0, 1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 7. Define the output directory
figs_dir = curr_dir / "results/figs/sloc_speedup_heatmap"
os.makedirs(figs_dir, exist_ok=True)

# 8. Adjust layout and save the figure
plt.tight_layout()
output_fig_path = figs_dir / f"{run_name}_heatmap_normalized.pdf"
plt.savefig(output_fig_path)

print(f"Heatmap figure saved to: {output_fig_path}")

# To display the plot in an interactive session (e.g., Jupyter)
# plt.show()
