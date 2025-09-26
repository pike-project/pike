import os
import subprocess
from pathlib import Path
from time import sleep
from datetime import datetime
import requests
import argparse


class SearchManager:
    def __init__(self, mode, worker_io_dir, port, level):
        self.mode = mode
        self.worker_io_dir = worker_io_dir
        self.port = port
        self.level = level
        self.curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

    def _start_openevolve_range(self, openevolve_dir, run_dir, root_dir, task_start, task_end):
        os.makedirs(run_dir, exist_ok=True)

        run_cmd = [
            "python",
            "examples/kernelbench/run.py",
            "--kernel_bench_dir",
            str(root_dir),
            "--level",
            self.level,
            "--task_start",
            str(task_start),
            "--task_end",
            str(task_end),
            "--eval_port",
            str(self.port),
            "--run_dir",
            str(run_dir),
        ]

        return subprocess.Popen(run_cmd, cwd=openevolve_dir)

    def _start_openevolve(self):
        openevolve_dir = Path("/global/scratch/users/knagaitsev/openevolve")
        example_dir = openevolve_dir / "examples/kernelbench"

        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = example_dir / "openevolve_output_runs" / timestamp_str
        os.makedirs(run_dir, exist_ok=True)

        root_dir = (self.curr_dir / "../..").resolve()

        run_ranges = [
            (1, 15),
            (16, 26),
            (27, 40),
            (41, 50),
        ]

        runs = []
        for r_lo, r_hi in run_ranges:
            run = self._start_openevolve_range(openevolve_dir, run_dir, root_dir, r_lo, r_hi)
            runs.append(run)

        sleep(2)
        return runs

    def _start_prev(self):
        # configs (lowercase)
        task_start = 1
        task_end = 50
        num_samples = 10
        num_phases = 30
        max_fix_attempts = 0
        dry_run = False
        server_type = "google"
        model_name = "gemini-2.5-pro"

        root_dir = (self.curr_dir / "../..").resolve()

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
            f"level={self.level}",
            f"task_start={task_start}",
            f"task_end={task_end}",
            f"num_samples={num_samples}",
            f"num_phases={num_phases}",
            f"max_fix_attempts={max_fix_attempts}",
            f"dry_run={str(dry_run)}",
            f"eval_port={self.port}",
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

    def run(self, use_openevolve=False):
        server_cmd = [
            "python", "scripts/disk_channel_server.py",
            "--port", str(self.port),
            "--worker_io_dir", str(self.worker_io_dir),
        ]

        disk_channel_server = subprocess.Popen(
            server_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        runs = []
        if use_openevolve:
            runs += self._start_openevolve()
        else:
            runs += self._start_prev()

        for run in runs:
            run.wait()

        requests.get(f"http://localhost:{self.port}/close")

        disk_channel_server.terminate()
        disk_channel_server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--worker_io_dir", type=str, required=False, default="worker_io")
    args = parser.parse_args()

    mode = args.mode
    worker_io_dir = args.worker_io_dir

    manager = SearchManager(mode, worker_io_dir, 8000, "3-metr")
    manager.run(use_openevolve=False)
