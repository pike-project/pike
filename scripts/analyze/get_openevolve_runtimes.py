
import os
import json
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_dir = Path("/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_runs/level3-metr_trial0")

tasks = []

for task_dirname in os.listdir(run_dir):
    task = int(task_dirname.split("task")[1])
    tasks.append(task)

tasks = sorted(tasks)

results = []

for task in tasks:
    task_dirname = f"task{task}"
    task_dir = run_dir / task_dirname

    best_info_path = task_dir / "output/best/best_program_info.json"

    with open(best_info_path) as f:
        best_info = json.load(f)
    
    runtime = best_info["metrics"]["runtime"]

    print(task, runtime)

    results.append({
        "problem_id": task,
        "runtime": runtime,
        # "results": {
        #     "eval_results": {
        #         "runtime": runtime,
        #     }
        # }
    })

out = {
    "title": "Ours (OpenEvolve)",
    "results": results,
}

output_path = (curr_dir / "../../results/breakdowns/a100_level3-metr/data/runtimes/ours_openevolve.json").resolve()

with open(output_path, "w") as f:
    json.dump(out, f)
