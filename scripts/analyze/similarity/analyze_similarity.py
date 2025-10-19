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

output_dir = (curr_dir / "../../../data/diffs" / run_name).resolve()
samples_dir = output_dir / "samples"
embeddings_dir = output_dir / "embeddings"

cos_sims = []

for task in sorted(os.listdir(embeddings_dir), key=lambda x: int(x.split("_")[1])):
    task_dir = embeddings_dir / task

    for sample in sorted(os.listdir(task_dir), key=lambda x: int(x.split("_")[1])):
        sample_dir = task_dir / sample

        emb1 = np.load(sample_dir / "seed.npy")
        emb2 = np.load(sample_dir / "code.npy")

        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        cos_sims.append(cos_sim)

sims_mean = np.mean(np.array(cos_sims))

print(f"Sims mean: {sims_mean}")
