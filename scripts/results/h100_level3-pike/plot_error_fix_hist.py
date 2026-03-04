import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Label map for display names ---
label_title_map = [
    ("prev_agents", "PIKE-B"),
    # ("openevolve_agents", "PIKE-O"),
    ("openevolve_agents_no_parallel_eval_no_islands", "PIKE-O (mut,npar,1isl)"),
]

is_mut = False
for (label, title) in label_title_map:
    if label == "openevolve_agents_no_parallel_eval_no_islands":
        is_mut = True
        break

# --- Setup paths ---
curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

data_dir = curr_dir / "results/data/error_fix_attempts"
figs_dir = curr_dir / "results/figs/error_fix_attempts"

os.makedirs(figs_dir, exist_ok=True)

fig_path = figs_dir / "hist.pdf"
if is_mut:
    fig_path = figs_dir / "hist_mut.pdf"

method_attempt_means = {}

# --- Load data ---
for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue

    label = filename.split(".")[0]
    with open(data_dir / filename) as f:
        data = json.load(f)

    means = []
    print(f"Task count: {len(data)}")
    for attempt_str, attempt_data in data.items():
        attempts_arr = []
        for (c, success) in attempt_data:
            attempts_arr.append(c)

        attempts_np = np.array(attempts_arr)
        mean_attempts = float(np.mean(attempts_np))
        means.append(mean_attempts)

    method_attempt_means[label] = means
    print(f"Found file: {label}")

# --- Plot histograms ---
if is_mut:
    plt.figure(figsize=(4, 2.25))
else:
    plt.figure(figsize=(4, 2))

# Define colors for consistency across histograms and mean lines
mean_colors = {
    "prev_agents": "blue",
    "openevolve_agents": "orange",
    "openevolve_agents_no_parallel_eval_no_islands": "orange",
}

# --- Compute a shared bin range for equal-width bins ---
all_means = np.concatenate(list(method_attempt_means.values()))
bin_edges = np.linspace(all_means.min(), all_means.max(), 21)  # 20 equal-width bins


# Add horizontal gridlines
plt.grid(axis='y', linestyle='--', alpha=0.5)

# --- Plot each method using shared bin edges ---
for label, title in label_title_map:
    means = method_attempt_means[label]

    # Plot histogram with shared bins
    n, bins, patches = plt.hist(
        means,
        bins=bin_edges,
        alpha=0.5,
        label=title,
        edgecolor="black"
    )

    print(n)

    if label == "openevolve_agents":
        # Suppose the last bar is the outlier (too tall)
        outlier_idx = np.argmax(n)
        true_height = n[outlier_idx]

        bar = patches[outlier_idx]
        x0 = bar.get_x()
        x1 = x0 + bar.get_width()

        plt.text((x0 + x1)/2, 7.05, f"{int(true_height)}", ha='center', va='bottom', fontsize=10, color='gray')

    # if label == "openevolve_agents_no_parallel_eval_no_islands":
    #     for patch in patches:
    #         patch.set_hatch('//')

    # --- Compute and plot mean of means as vertical dashed line ---
    mean_of_means = np.mean(means)
    plt.axvline(mean_of_means, color=mean_colors[label], linestyle='--', linewidth=1)

plt.xlim(1, 5)
plt.ylim(0, 7)

plt.xticks(range(1, 6, 1))

plt.xlabel("(a) Mean Error Fix Attempts")
plt.ylabel("Tasks")
# plt.title("Mean Task Attempts Required")

if is_mut:
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.46, 1.02),
        ncol=2,
        frameon=True
    )
else:
    plt.legend()

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

# --- Save the figure ---
plt.savefig(fig_path, format="pdf")
plt.close()

print(f"Histogram saved to {fig_path}")
