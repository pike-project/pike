import os
import json
import difflib
import shutil
import subprocess
from pathlib import Path
import numpy as np
from radon.raw import analyze

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# run_name = "h100_level_3-metr_prev_agents_trial_1"
run_name = "h100_level_3-metr_openevolve_agents_trial_0"

output_dir = (curr_dir / "../../../data/diffs" / run_name).resolve()
samples_dir = output_dir / "samples"
embeddings_dir = output_dir / "embeddings"

task_means = []

for task in sorted(os.listdir(samples_dir), key=lambda x: int(x.split("_")[1])):
    code_lens = []
    
    task_dir = samples_dir / task

    for sample in sorted(os.listdir(task_dir), key=lambda x: int(x.split("_")[1])):
        sample_dir = task_dir / sample

        with open(sample_dir / "seed.py") as f:
            seed = f.read()
        
        with open(sample_dir / "code.py") as f:
            code = f.read()

        # code_len = len(code.split("\n"))
        try:
            code_len = analyze(code).sloc
        except Exception as e:
            print(f"Task {task} sample {sample} failed with radon sloc")
            code_len = len(code.split("\n"))

        code_lens.append(code_len)

    task_mean = np.mean(np.array(code_lens))

    task_means.append(task_mean)

mean_of_means = np.mean(np.array(task_means))


print(f"Code length mean of means: {mean_of_means}")
