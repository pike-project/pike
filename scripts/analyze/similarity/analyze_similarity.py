import os
import json
import difflib
import shutil
import subprocess
from pathlib import Path
import numpy as np

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_name = "h100_level_3-metr_prev_agents_trial_1"
# run_name = "h100_level_3-metr_openevolve_agents_trial_0"

target_level = "3-metr"

task_blacklist_map = {
    "5": set(),
    "3-metr": {
        36,
        37,
        38,
        39,
        40,
        41,
        42,
    },
}

task_blacklist = task_blacklist_map.get(target_level, set())

output_dir = (curr_dir / "../../../data/diffs" / run_name).resolve()
samples_dir = output_dir / "samples"
embeddings_dir = output_dir / "embeddings"

task_means = []

for task in sorted(os.listdir(embeddings_dir), key=lambda x: int(x.split("_")[1])):
    task_num = int(task.split("_")[1].split(".py")[0])
    if task_num in task_blacklist:
        continue

    print(f"Task: {task_num}")

    cos_sims = []
    
    task_dir = embeddings_dir / task

    for sample in sorted(os.listdir(task_dir), key=lambda x: int(x.split("_")[1])):
        sample_dir = task_dir / sample

        emb1 = np.load(sample_dir / "seed.npy")
        emb2 = np.load(sample_dir / "code.npy")

        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        cos_sims.append(cos_sim)

    sims_mean = np.mean(np.array(cos_sims))

    task_means.append(sims_mean)

mean_of_means = np.mean(np.array(task_means))

print(f"Cosine similarity mean of means: {mean_of_means}")
