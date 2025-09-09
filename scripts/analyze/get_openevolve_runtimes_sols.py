
import os
import json
import shutil
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

run_dir = Path("/pscratch/sd/k/kir/llm/openevolve/examples/kernelbench/openevolve_output_lrc/h100_level_3-metr_trial_0")

sol_dest_dir = (curr_dir / "../../best_agent_solutions/h100/level3-metr/openevolve_no_agents_300/best_solutions").resolve()
os.makedirs(sol_dest_dir, exist_ok=True)

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

    sol_src = task_dir / "output/best/best_program.py"
    sol_dest = sol_dest_dir / f"task_{task}.py"

    shutil.copy(sol_src, sol_dest)

out = {
    "title": "Ours (OpenEvolve)",
    "results": results,
}

output_path = (curr_dir / "../../results/breakdowns/h100_level3-metr/data/runtimes/ours_openevolve.json").resolve()

with open(output_path, "w") as f:
    json.dump(out, f)
