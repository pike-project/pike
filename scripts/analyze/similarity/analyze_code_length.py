import os
import json
import difflib
import shutil
import subprocess
from pathlib import Path
import numpy as np
from radon.raw import analyze
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# run_name = "h100_level_3-metr_prev_agents_trial_1"
run_name = "h100_level_3-metr_openevolve_agents_trial_0"
# run_name = "h100_level_3-metr_openevolve_noagents_trial_0"

target_level = "3-metr"

analyze_working = True

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

if analyze_working:
    output_dir = (curr_dir / "../../../data/diffs_working" / run_name).resolve()
else:
    output_dir = (curr_dir / "../../../data/diffs" / run_name).resolve()


def get_baseline_runtime(data, task):
    for v in data["results"]:
        if v["problem_id"] == task:
            try:
                return v["results"]["eval_results"]["runtime"]
            except Exception as e:
                return None
    
    return None

eager_path = (curr_dir / "../../../results/ours/h100_level3-metr/results/data/runtimes/eager.json").resolve()

with open(eager_path) as f:
    eager_runtimes = json.load(f)

samples_dir = output_dir / "samples"
embeddings_dir = output_dir / "embeddings"

task_means = []

output_data = []

for task in sorted(os.listdir(samples_dir), key=lambda x: int(x.split("_")[1])):
    task_num = int(task.split("_")[1].split(".py")[0])
    if task_num in task_blacklist:
        continue

    # if task_num != 10:
    #     continue

    eager_runtime = get_baseline_runtime(eager_runtimes, task_num)

    speedups = []
    runtimes = []
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

        if analyze_working:
            with open(sample_dir / "runtime.txt") as f:
                runtime = float(f.read())
            
            runtimes.append(runtime)
            speedup = eager_runtime / runtime
            speedups.append(speedup)
            
            print(f"SLOC: {code_len}, runtime: {runtime}")
        
        # if code_len > 470:
        #     with open("big_code.py", "w") as f:
        #         f.write(code)
        
        # if speedup > 1.2:
        #     with open("fast_code.py", "w") as f:
        #         f.write(code)

    task_mean = np.mean(np.array(code_lens))

    task_means.append(task_mean)

    print(f"{task} mean SLOC: {task_mean}")

    # plt.figure()
    # plt.scatter(code_lens, speedups)
    # plt.xlabel("SLOC")
    # plt.ylabel("Speedup")

    # figs_dir = curr_dir / "../../../results/ours/h100_level3-metr/results/figs/sloc_speedup_scatter/tasks"

    # os.makedirs(figs_dir, exist_ok=True)

    # plt.savefig(figs_dir / f"task_{task_num}.pdf")

    output_data.append({
        "task": task_num,
        "slocs": code_lens,
        "runtimes": runtimes,
        "speedups": speedups,
    })

if analyze_working:
    output_data_path = (curr_dir / f"../../../results/ours/h100_level3-metr/results/data/sloc_speedup/{run_name}.json").resolve()

    with open(output_data_path, "w") as f:
        json.dump(output_data, f, indent=4)

mean_of_means = np.mean(np.array(task_means))

print(f"Code length mean of means: {mean_of_means}")
