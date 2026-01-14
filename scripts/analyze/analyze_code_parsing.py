import os
import json
from collections import defaultdict
from pathlib import Path
from src.utils import extract_first_code

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

target_attempt = 300

# run_name = "h100_level_3-metr_prev_agents_cheap_efa_0"
run_name = "2026_01_13_20_22_23"

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

root_dir = (curr_dir / "../../data/parallel_runs" / run_name / "runs/runs/run_0/run/tasks").resolve()

attempt_count = 0
kernel_good_count = 0
kernel_none_count = 0
res_missing_count = 0

for task_name in os.listdir(root_dir):
    if not task_name.startswith("task"):
        continue
    task = int(task_name.split("task")[1])

    if task in task_blacklist:
        continue

    task_path = os.path.join(root_dir, task_name)
    if not os.path.isdir(task_path):
        continue

    iter_output_dir = os.path.join(task_path, "output", "iter_output")
    if not os.path.exists(iter_output_dir):
        continue

    iter_names = os.listdir(iter_output_dir)
    # Sort the iteration directories numerically to process them in order
    iter_names_sorted = sorted(
        [d for d in iter_names if d.startswith("iter_")],
        key=lambda x: int(x.split("_")[1])
    )

    for iter_name in iter_names_sorted:
        iter_path = os.path.join(iter_output_dir, iter_name)
        if not os.path.isdir(iter_path):
            continue

        attempts_dir = Path(os.path.join(iter_path, "attempts"))
        if not os.path.exists(attempts_dir):
            continue

        attempt_dirnames = sorted(os.listdir(attempts_dir), key=lambda x: int(x.split("_")[1]))

        for attempt_dirname in attempt_dirnames:
            attempt_dir = attempts_dir / attempt_dirname

            attempt_count += 1

            res_path = attempt_dir / "response.md"

            try:
                with open(res_path) as f:
                    raw_res = f.read()
            except Exception as e:
                res_missing_count += 1
                continue

            custom_kernel = extract_first_code(raw_res, ["python", "cpp"])

            if custom_kernel is None:
                kernel_none_count += 1
            else:
                kernel_good_count += 1
                # print(res_path)

assert attempt_count == res_missing_count + kernel_none_count + kernel_good_count, "Counts must match"

print(f"Total attempt count: {attempt_count}, res missing: {res_missing_count}, kernel none count: {kernel_none_count}, kernel good count: {kernel_good_count}")
