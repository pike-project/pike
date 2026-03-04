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
    slocs = np.array(v["slocs"])
    speedups = np.array(v["speedups"])

    # Select the current subplot axis
    ax = axes[i]
    
    # --- START: NORMALIZATION ---
    # Skip plotting if there's no data or not enough to form a range
    if len(slocs) < 2 or len(speedups) < 2:
        ax.set_title(f"Task: {task} (Not enough data)")
        ax.axis('off')
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
    # --- END: NORMALIZATION ---


    # 5. Plot the NORMALIZED data on the specific subplot axis (ax)
    ax.scatter(normalized_slocs, normalized_speedups)

    # --- START: MODIFIED LINEAR FIT ON NORMALIZED DATA ---
    # Calculate the coefficients (slope m, intercept b) using normalized data
    # m, b = np.polyfit(normalized_slocs, normalized_speedups, 1)
    
    # # Create a line across the normalized range [0, 1]
    # x_fit = np.array([0, 1])
    # y_fit = m * x_fit + b
    
    # Plot the linear fit line
    # ax.plot(x_fit, y_fit, color='green', linestyle='-', linewidth=2, label='Linear Fit')
    # --- END: MODIFIED LINEAR FIT ---

    # --- START: MODIFIED BASELINE (y=1) ---
    # Normalize the position of the original y=1 line
    # Only draw the line if the original speedups actually spanned the value 1.
    # if min_speedup < 1 < max_speedup:
    #     normalized_y1 = (1 - min_speedup) / (max_speedup - min_speedup)
    #     ax.axhline(y=normalized_y1, color='red', linestyle='--', linewidth=1.2)
    # --- END: MODIFIED BASELINE ---

    ax.set_title(f"Task: {task}")
    ax.set_xlabel("Normalized SLOC")
    ax.set_ylabel("Normalized Speedup")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set axis limits to be explicitly [0, 1] for clarity
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


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
output_fig_path = figs_dir / f"{run_name}_subplots_normalized.pdf"
plt.savefig(output_fig_path)

print(f"Normalized subplots figure saved to: {output_fig_path}")

# To display the plot in an interactive session (e.g., Jupyter)
# plt.show()
