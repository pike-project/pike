import os
import subprocess
from pathlib import Path
from time import sleep
from datetime import datetime
import requests

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

# Need to run: python scripts/disk_channel_server.py --port 8000

# Then need to run n of these in parallel, splitting up the task range:
# python examples/kernelbench/run.py --kernel_bench_dir /global/scratch/users/knagaitsev/KernelBench --level 3-metr --task_start 10 --task_end 20 --eval_port 8000

# Once all n of those are finished, need to:
# - kill disk_channel_server
# - send close message to the eval worker to end the H100 job (should we kill the slurm job instead?)

def main():
    port = 8000

    disk_channel_server = subprocess.Popen(
        ["python", "scripts/disk_channel_server.py", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # openevolve_dir = Path("/pscratch/sd/k/kir/llm/openevolve")
    openevolve_dir = Path("/global/scratch/users/knagaitsev/openevolve")

    example_dir = openevolve_dir / "examples/kernelbench"

    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = example_dir / "openevolve_output_runs" / timestamp_str
    os.makedirs(run_dir, exist_ok=True)

    root_dir = (curr_dir / "../..").resolve()

    level = "3-metr"
    task_start = 1
    task_end = 1

    run_cmd = [
        "python",
        "examples/kernelbench/run.py",
        "--kernel_bench_dir",
        str(root_dir),
        "--level",
        level,
        "--task_start",
        task_start,
        "--task_end",
        task_end,
        "--eval_port",
        str(port),
        "--run_dir",
        run_dir,
    ],

    run = subprocess.Popen(
        run_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=openevolve_dir,
    )

    run.wait()

    requests.get(f"http://localhost:{port}/close")

    disk_channel_server.terminate()
    disk_channel_server.wait()

if __name__ == "__main__":
    main()
