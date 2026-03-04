import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path


curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_name = "h100_level_3-metr_prev_agents_trial_1"
output_data_path = (curr_dir / f"results/data/sloc_speedup/{run_name}.json").resolve()

with open(output_data_path) as f:
    data = json.load(f)

# --- Step 1: Compute simple feature vectors for clustering ---
# Each task becomes a row in the feature matrix
features = []
for v in data:
    slocs = np.array(v["slocs"])
    speedups = np.array(v["speedups"])
    features.append([
        np.mean(slocs),
        np.std(slocs),
        np.mean(speedups),
        np.std(speedups),
        np.corrcoef(slocs, speedups)[0, 1],  # relationship between sloc & speedup
    ])
features = np.nan_to_num(np.array(features))  # handle NaN correlations safely

# --- Step 2: Cluster tasks based on these features ---
n_clusters = 4  # you can tune this or use silhouette score to choose automatically
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features)

# --- Step 3: Plot all data on a single combined scatter plot ---
plt.figure(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for i, v in enumerate(data):
    slocs = np.array(v["slocs"])
    speedups = np.array(v["speedups"])
    cluster_id = clusters[i]
    plt.scatter(
        slocs, speedups,
        color=colors[cluster_id],
        alpha=0.6,
        label=f"Cluster {cluster_id}" if f"Cluster {cluster_id}" not in plt.gca().get_legend_handles_labels()[1] else None
    )

plt.axhline(y=1, color='red', linestyle='--', linewidth=1.2)
plt.xlabel("SLOC")
plt.ylabel("Speedup")
plt.title("Merged SLOC vs Speedup (Clustered Tasks)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Clusters")
plt.tight_layout()

# --- Save merged figure ---
figs_dir = curr_dir / "results/figs/sloc_speedup_scatter"
os.makedirs(figs_dir, exist_ok=True)
output_fig_path = figs_dir / f"{run_name}_merged_clustered.pdf"
plt.savefig(output_fig_path)
print(f"Clustered merged plot saved to: {output_fig_path}")
# plt.show()
