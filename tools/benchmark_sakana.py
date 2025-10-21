import os, sys
from pathlib import Path
import json

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

kernel_bench_dir = (curr_dir / "../KernelBench").resolve()

level_dir = kernel_bench_dir / "level3-metr"

level_num = 3

task_nums = []

for filename in os.listdir(level_dir):
    if not filename.endswith(".py"):
        continue

    task_nums.append(int(filename.split("_")[0]))

task_nums = sorted(task_nums)

results = []

for task_num in task_nums:
    output_path = f"/scratch/task_{task_num}.json"
    # TODO: execute this
    # python scripts/eval.py --level {level_num} --task {task_num} --code_path local/deps/KernelBench-analysis/best/level_{level_num}/task_{task_num}/pytorch_functional.py --output_path {output_path} --cuda_path local/deps/KernelBench-analysis/best/level_{level_num}/task_{task_num}/kernel.cu

    try:
        with open(output_path) as f:
            data = json.load(f)
        
        runtime = data["llm"]["runtime"]

        results.append({
            "problem_id": task_num,
            "runtime": runtime,
            "results": {
                "eval_results": data["llm"],
            },
        })
    except Exception as e:
        print(e)

full_output = {
    "title": "Sakana",
    "results": results,
}

with open(f"/output/sakana.json") as f:
    json.dump(full_output, f)
