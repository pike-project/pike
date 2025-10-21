import os, sys, subprocess
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
    cmd = [
        "python",
        "scripts/eval.py",
        "--level", str(level_num),
        "--task", str(task_num),
        "--code_path", f"local/deps/KernelBench-analysis/best/level_{level_num}/task_{task_num}/pytorch_functional.py",
        "--output_path", output_path,
        "--cuda_path", f"local/deps/KernelBench-analysis/best/level_{level_num}/task_{task_num}/kernel.cu",
    ]

    print(f"\n>>> Running command for task {task_num}: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        print(f"Command failed for task {task_num}: {e}")

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

        print(f"Task {task_num} completed successfully. Runtime: {runtime}")

    except Exception as e:
        print(f"Failed to process output for task {task_num}: {e}")

full_output = {
    "title": "Sakana",
    "results": results,
}

os.makedirs("/output", exist_ok=True)
with open("/output/sakana.json", "w") as f:
    json.dump(full_output, f, indent=2)

print("\nAll done. Results written to /output/sakana.json")
