import os
import subprocess
from pathlib import Path
from time import sleep
from datetime import datetime
import requests

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def start_openevolve_range(openevolve_dir, run_dir, root_dir, port, level, task_start, task_end):
    os.makedirs(run_dir, exist_ok=True)

    run_cmd = [
        "python",
        "examples/kernelbench/run.py",
        "--kernel_bench_dir",
        str(root_dir),
        "--level",
        level,
        "--task_start",
        str(task_start),
        "--task_end",
        str(task_end),
        "--eval_port",
        str(port),
        "--run_dir",
        str(run_dir),
    ]

    run = subprocess.Popen(
        run_cmd,
        cwd=openevolve_dir,
    )

    return run

def start_openevolve(port):
    openevolve_dir = Path("/global/scratch/users/knagaitsev/openevolve")
    example_dir = openevolve_dir / "examples/kernelbench"

    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = example_dir / "openevolve_output_runs" / timestamp_str
    os.makedirs(run_dir, exist_ok=True)

    root_dir = (curr_dir / "../..").resolve()
    level = "3-metr"

    run_ranges = [
        (1, 15),
        (16, 26),
        (27, 40),
        (41, 50),
    ]

    runs = []
    for (r_lo, r_hi) in run_ranges:
        run = start_openevolve_range(openevolve_dir, run_dir, root_dir, port, level, r_lo, r_hi)
        runs.append(run)

    sleep(2)
    return runs

def start_prev(port):
    # configs (lowercase)
    level = "3-metr"
    task_start = 1
    task_end = 50
    num_samples = 10
    num_phases = 30
    max_fix_attempts = 0
    dry_run = False
    server_type = "google"
    model_name = "gemini-2.5-pro"

    root_dir = (curr_dir / "../..").resolve()

    data_dir = (root_dir / "data").resolve()
    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = data_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "out.log"

    run_cmd = [
        "python", "-u", "scripts/parallel_tree_search.py",
        f"run_dir={run_dir}",
        f"server_type={server_type}",
        f"model_name={model_name}",
        "num_workers=30",
        f"level={level}",
        f"task_start={task_start}",
        f"task_end={task_end}",
        f"num_samples={num_samples}",
        f"num_phases={num_phases}",
        f"max_fix_attempts={max_fix_attempts}",
        f"dry_run={str(dry_run)}",
        f"eval_port={port}",
    ]

    log_file = open(log_path, "a")
    run = subprocess.Popen(
        run_cmd,
        cwd=root_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    sleep(2)
    return [run]

def main():
    port = 8000

    disk_channel_server = subprocess.Popen(
        ["python", "scripts/disk_channel_server.py", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # cwd=curr_dir,
    )

    runs = []
    # runs += start_openevolve(port)
    runs += start_prev(port)

    for run in runs:
        run.wait()

    requests.get(f"http://localhost:{port}/close")

    disk_channel_server.terminate()
    disk_channel_server.wait()

if __name__ == "__main__":
    main()
