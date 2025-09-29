import os
from pathlib import Path
import subprocess
import argparse
import shutil
from datetime import datetime
from time import sleep
import signal

"""
The parallel manager does the following:
- Create a fresh data/parallel_runs dir
    - Create runs dir within it
    - Create a pending_gpu_allocations dir and poll on it. The moment that the worker starts, a file will be written to this directory
        telling the parallel manager that the allocation has been obtained
- Get a large GPU allocation for the evaluator (8 H100s, 112 CPUs, 72 hours, 56 parallel jobs running)
    - worker_io for the worker needs to all be placed within the parallel run dir, to ensure full jobs can be separated out (if we have multiple GPU allocations)
- Get a CPU allocation the moment the GPU allocation is ready
"""

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

class ParallelManager:
    def __init__(self):
        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        parent_run_dir = (curr_dir / "../../data/parallel_runs" / timestamp_str).resolve()

        run_dir = parent_run_dir / "runs" 
        worker_io_dir = parent_run_dir / "worker_io"

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(worker_io_dir, exist_ok=True)

        self.run_dir = run_dir

        self.worker_io_dir = worker_io_dir

    # blocks until the worker is ready
    def start_eval_worker(self):
        worker_script_path = (curr_dir / "../worker_jobs/lrc.py").resolve()

        cmd = [
            "python",
            worker_script_path,
            "--worker_io_dir",
            self.worker_io_dir,
            "--gpu_count", 2,
            "--cpu_count", 40,
            "--max_active_tasks", 20,
            "--allocation_time", "24:00:00",
        ]
        cmd = [str(x) for x in cmd]

        with open(self.run_dir / "worker.log", "w") as f:
            worker = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
            )

        worker_ready_path = self.worker_io_dir / "ready.txt"

        while True:
            try:
                if os.path.isfile(worker_ready_path):
                    break
            except Exception as e:
                print(e)
                continue

        return worker

    def start_search(self):
        search_manager_path = (curr_dir / "search_manager.py").resolve()

        cmd = [
            "srun",
            "--account=ac_binocular",
            "--partition=lr8",
            "--mincpus=64",
            "--mem=64G",
            "--nodes=1",
            "--qos=lr8_normal",
            "--time=72:0:0",
            "python",
            "-u",
            str(search_manager_path),
            "--worker_io_dir",
            str(self.worker_io_dir),
            "--mode", "prev_noagents",
            "--run_dir", str(self.run_dir),
            "--level", "0",
        ]

        with open(self.run_dir / "search.log", "w") as f:
            search_proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
            )

        return search_proc

    def run(self):
        print("Starting worker, waiting for it to be ready...")

        worker = self.start_eval_worker()

        sleep(10)

        print("Worker ready! Starting search.")

        search = self.start_search()
        search.wait()

        worker.terminate()
        worker.wait()

        print("Search complete, worker terminated.")

if __name__ == "__main__":
    pm = ParallelManager()
    pm.run()
