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

# --- Plotting Modifications ---

# 1. Define the grid dimensions
rows = 5
cols = 6
num_subplots = rows * cols

# 2. Create a single figure and a grid of subplots before the loop
#    `figsize` is adjusted to make the large plot readable.
fig, axes = plt.subplots(rows, cols, figsize=(24, 20))

# 3. Flatten the 2D array of axes to make it easy to iterate over
axes = axes.flatten()

# 4. Loop through the data and the axes simultaneously
for i, v in enumerate(data):
    # Stop if we have more data than subplots
    if i >= num_subplots:
        print(f"Warning: Data contains more items ({len(data)}) than available subplots ({num_subplots}). Stopping.")
        break

    task = v["task"]
    slocs = v["slocs"]
    speedups = v["speedups"]

    # Select the current subplot axis
    ax = axes[i]

    # 5. Plot on the specific subplot axis (ax) instead of the global `plt`
    ax.scatter(slocs, speedups)

    # --- START: ADDED LINEAR FIT ---
    # Ensure there are at least two points to fit a line
    if len(slocs) > 1:
        # Convert to numpy arrays for calculation
        x = np.array(slocs)
        y = np.array(speedups)
        
        # Calculate the coefficients (slope m, intercept b) of the linear fit
        m, b = np.polyfit(x, y, 1)
        
        # Create a smooth line for plotting using the min and max of the x-data
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = m * x_fit + b
        
        # Plot the linear fit line
        ax.plot(x_fit, y_fit, color='green', linestyle='-', linewidth=2, label='Linear Fit')
    # --- END: ADDED LINEAR FIT ---

    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.2)

    ax.set_title(f"Task: {task}")
    ax.set_xlabel("SLOC")
    ax.set_ylabel("Speedup")
    ax.grid(True, linestyle='--', alpha=0.6)


# 6. Hide any unused subplots at the end
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# 7. Define the output directory for the combined figure
#    Saving to a different directory to avoid confusion with the single task plots
figs_dir = curr_dir / "results/figs/sloc_speedup_scatter"
os.makedirs(figs_dir, exist_ok=True)

# 8. Adjust layout to prevent titles and labels from overlapping
plt.tight_layout()

# 9. Save the entire figure to a single file, outside the loop
output_fig_path = figs_dir / f"{run_name}_subplots.pdf"
plt.savefig(output_fig_path)

print(f"Subplots figure saved to: {output_fig_path}")

# To display the plot in an interactive session (e.g., Jupyter)
# plt.show()
