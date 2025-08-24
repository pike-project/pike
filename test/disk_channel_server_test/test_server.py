import requests
import os
from pathlib import Path
from time import sleep

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

base_url = "http://localhost:8000"

level = 0
task = 1

level_dir = (curr_dir / f"../../KernelBench/level{level}").resolve()

task_path = None
for filename in os.listdir(level_dir):
    if not filename.endswith(".py"):
        continue

    file_task = int(filename.split("_")[0])
    if task == file_task:
        task_path = level_dir / filename
        break

with open(task_path) as f:
    code = f.read()

submit_path = "/submit"
submit_params = {
    "code": code,
    "level": level,
    "task": task
}

res = requests.get(f"{base_url}{submit_path}", params=submit_params)

eval_id = res.text

poll_path = "/poll"
poll_params = {"id": eval_id}

while True:
    res = requests.get(f"{base_url}{poll_path}", params=poll_params)
    data = res.json()
    print(data)
    if data is not None:
        break

    sleep(5)
