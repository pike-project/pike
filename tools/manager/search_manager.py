import os
import subprocess
from pathlib import Path
from time import sleep
from datetime import datetime
import requests
import argparse
from math import ceil
import time

def split_into_ranges(nums, n_ranges):
    if not nums:
        return []
    nums = sorted(set(nums))
    total = len(nums)
    if n_ranges <= 0:
        raise ValueError("n_ranges must be > 0")
    # keep your original edge-case behavior: if you request >= total ranges,
    # return one range per number
    if n_ranges >= total:
        return [(x, x) for x in nums]
    
    base = total // n_ranges
    remainder = total % n_ranges  # first `remainder` ranges get one extra element
    ranges = []
    idx = 0
    for i in range(n_ranges):
        size = base + (1 if i < remainder else 0)
        start_val = nums[idx]
        end_val = nums[idx + size - 1]
        idx += size
        # expand end to fill the gap up to just before the next group's start
        if i < n_ranges - 1:
            next_start = nums[idx]
            end_val = next_start - 1
        ranges.append((start_val, end_val))
    return ranges


class SearchManager:
    def __init__(self, mode, worker_io_dir, run_dir, port, level, run_count, range_count):
        self.mode = mode
        self.use_agents = mode.split("_")[1] == "agents"

        self.worker_io_dir = worker_io_dir
        self.port = port
        self.level = level
        self.curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
        self.root_dir = (self.curr_dir / "../..").resolve()
        self.run_count = run_count

        self.curr_run_count = 0
        self.curr_partition_id = 0

        self.range_count = range_count

        if run_dir is None:
            data_dir = (self.root_dir / "data").resolve()
            run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            run_dir = data_dir / "runs" / run_name
            run_dir.mkdir(exist_ok=True)

            self.parent_run_dir = run_dir
        else:
            self.parent_run_dir = Path(run_dir)

        self.ranges = self.compute_ranges()
        print(f"Using run ranges: {self.ranges}")

    def create_run_dir_log_dir(self):
        root_run_dir = self.parent_run_dir / "runs" / f"run_{self.curr_run_count}"

        self.curr_run_count += 1
        self.curr_partition_id = 0

        log_dir = root_run_dir / "logs"
        run_dir = root_run_dir / "run"

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)

        return run_dir, log_dir

    def _start_openevolve_range(self, openevolve_dir, run_dir, log_dir, task_start, task_end):
        os.makedirs(run_dir, exist_ok=True)

        run_cmd = [
            "python",
            "examples/kernelbench/run.py",
            "--kernel_bench_dir",
            str(self.root_dir),
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

        log_path = log_dir / f"logs_{self.curr_partition_id}.log"
        self.curr_partition_id += 1
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                run_cmd,
                cwd=openevolve_dir,
                stdout=f,
                stderr=f,
            )

        return proc

    def _start_openevolve(self):
        openevolve_dir = Path("/global/scratch/users/knagaitsev/openevolve")
        example_dir = openevolve_dir / "examples/kernelbench"

        run_dir, log_dir = self.create_run_dir_log_dir()

        run_ranges = self.ranges

        runs = []
        for r_lo, r_hi in run_ranges:
            run = self._start_openevolve_range(openevolve_dir, run_dir, log_dir, r_lo, r_hi)
            runs.append(run)

        return runs

    def _start_prev_range(self, run_dir, log_dir, task_start, task_end):
        os.makedirs(run_dir, exist_ok=True)

        # configs
        num_samples = 10
        num_phases = 30
        max_fix_attempts = 5 if self.use_agents else 0
        dry_run = False
        server_type = "google"
        model_name = "gemini-2.5-pro"

        run_cmd = [
            "python", "-u", "scripts/parallel_tree_search.py",
            f"run_dir={run_dir}",
            f"server_type={server_type}",
            f"model_name={model_name}",
            "num_workers=10",
            f"level={self.level}",
            f"task_start={task_start}",
            f"task_end={task_end}",
            f"num_samples={num_samples}",
            f"num_phases={num_phases}",
            f"max_fix_attempts={max_fix_attempts}",
            f"dry_run={str(dry_run)}",
            f"eval_port={self.port}",
        ]

        log_path = log_dir / f"logs_{self.curr_partition_id}.log"
        self.curr_partition_id += 1
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                run_cmd,
                cwd=self.root_dir,
                stdout=f,
                stderr=f,
            )

        return proc

    def _start_prev(self):
        run_dir, log_dir = self.create_run_dir_log_dir()

        run_ranges = self.ranges

        runs = []
        for r_lo, r_hi in run_ranges:
            run = self._start_prev_range(run_dir, log_dir, r_lo, r_hi)
            runs.append(run)

        return runs

    def run(self):
        server_cmd = [
            "python", "scripts/disk_channel_server.py",
            "--port", str(self.port),
            "--worker_io_dir", str(self.worker_io_dir),
        ]

        # with open(self.parent_run_dir / "disk_channel_server.log", "w") as f:
        #     disk_channel_server = subprocess.Popen(
        #         server_cmd,
        #         stdout=f,
        #         stderr=f,
        #     )
        disk_channel_server = subprocess.Popen(
            server_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(self.run_count):
            runs = []

            start_time = time.time()

            if self.mode == "openevolve_agents" or self.mode == "openevolve_noagents":
                runs += self._start_openevolve()
            else:
                runs += self._start_prev()

            for run in runs:
                run.wait()
            
            end_time = time.time()
            run_time = end_time - start_time

            print(f"Time to complete run: {run_time:.2f}s")

        sleep(10)

        requests.get(f"http://localhost:{self.port}/close")

        disk_channel_server.terminate()
        disk_channel_server.wait()


    def compute_ranges(self):
        range_count = self.range_count

        tasks = []

        level_dir = self.root_dir / f"KernelBench/level{self.level}"
        for filename in os.listdir(level_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[0])
            tasks.append(task)

        tasks = sorted(tasks)

        return split_into_ranges(tasks, range_count)

def test_split_into_ranges():
    # nums = [1, 5, 14, 16, 17, 26, 28, 33, 34, 42, 50]
    # result = split_into_ranges(nums, 4)
    # print(result)

    manager = SearchManager("prev_noagents", "worker_io", "runs/tmp", 8000, "3-metr", 1, 36)
    print(manager.ranges)
    assert manager.ranges == [
        (1, 3), (4, 5), (6, 7), (8, 8), (9, 9), (10, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 18),
        (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 28), (29, 29),
        (30, 30), (31, 32), (33, 35), (36, 36), (37, 37), (38, 38), (39, 39), (40, 40), (41, 41), (42, 42),
        (43, 43), (44, 47), (48, 48), (49, 49), (50, 50)
    ], "ranges incorrect for level 3-metr, 36 ranges"

    manager = SearchManager("prev_noagents", "worker_io", "runs/tmp", 8000, "3-metr", 1, 37)
    print(manager.ranges)
    assert manager.ranges == [
        (1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (9, 9), (10, 10), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17),
        (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (29, 29), (30, 30),
        (31, 31), (33, 33), (36, 36), (37, 37), (38, 38), (39, 39), (40, 40), (41, 41), (42, 42), (43, 43), (44, 44),
        (48, 48), (49, 49), (50, 50)
    ], "ranges incorrect for level 3-metr, 37 ranges"

    manager = SearchManager("prev_noagents", "worker_io", "runs/tmp", 8000, "3-metr", 1, 50)
    print(manager.ranges)
    assert manager.ranges == [
        (1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (9, 9), (10, 10), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17),
        (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (29, 29), (30, 30),
        (31, 31), (33, 33), (36, 36), (37, 37), (38, 38), (39, 39), (40, 40), (41, 41), (42, 42), (43, 43), (44, 44),
        (48, 48), (49, 49), (50, 50)
    ], "ranges incorrect for level 3-metr, 50 ranges"

    manager = SearchManager("prev_noagents", "worker_io", "runs/tmp", 8000, "3-metr", 1, 4)
    print(manager.ranges)
    assert manager.ranges == [(1, 15), (16, 25), (26, 38), (39, 50)], "ranges incorrect for level 3-metr, 4 ranges"

    manager = SearchManager("prev_noagents", "worker_io", "runs/tmp", 8000, "3-metr", 1, 5)
    print(manager.ranges)
    assert manager.ranges == [(1, 13), (14, 22), (23, 30), (31, 40), (41, 50)], "ranges incorrect for level 3-metr, 5 ranges"

    manager = SearchManager("prev_noagents", "worker_io", "runs/tmp", 8000, "5", 1, 5)
    print(manager.ranges)
    assert manager.ranges == [(1, 3), (4, 6), (7, 9), (10, 12), (13, 14)], "ranges incorrect for level 5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=False, default="3-metr")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--worker_io_dir", type=str, required=False, default="worker_io")
    parser.add_argument("--run_dir", type=str, required=False, default=None)
    parser.add_argument("--run_count", type=int, required=False, default=1)
    parser.add_argument("--ranges", type=int, required=False, default=50)
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()

    if args.test:
        test_split_into_ranges()
        exit(0)

    valid_modes = {
        "openevolve_agents",
        "openevolve_noagents",
        "prev_agents",
        "prev_noagents",
    }

    if args.mode not in valid_modes:
        raise Exception("Provided mode not in valid modes: {args.mode}, valid modes are: {valid_modes}")

    mode = args.mode

    worker_io_dir = Path(args.worker_io_dir)

    run_ranges = args.ranges

    manager = SearchManager(mode, worker_io_dir, args.run_dir, 8000, args.level, args.run_count, run_ranges)

    manager.run()
