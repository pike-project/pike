import random
import os, sys
from pathlib import Path
import asyncio
import uuid
import json
import argparse
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.util.disk_channel import DiskChannel

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

class EvalSolutions:
    def __init__(self, level: int, mode: str, solutions_name: str, run_name: str, results_dir: Path, worker_input_dir: Path, worker_output_dir: Path, dry_run: bool):
        tx_dir = worker_input_dir
        rx_dir = worker_output_dir

        print(tx_dir, rx_dir)

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.results_dir = results_dir

        self.dup_count = 1

        self.level = level
        self.mode = mode
        self.solutions_name = solutions_name
        self.run_name = run_name
        self.dry_run = dry_run
    
    def metr_solutions(self):
        kernel_bench_dir = (curr_dir / "../../../KernelBenchFiltered").resolve()

        level_dir = kernel_bench_dir / f"best_agent_solutions/level_{self.level}"

        tasks = []

        for filename in os.listdir(level_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[1].split(".py")[0])
            
            file_path = level_dir / filename

            with open(file_path) as f:
                code = f.read()
            
            for _ in range(self.dup_count):
                tasks.append({
                    "code": code,
                    "problem_id": task
                })

        tasks.sort(key=lambda x: x["problem_id"])

        for idx, task in enumerate(tasks):
            task["sample_id"] = idx

        return tasks

    def ground_truth_solutions(self):
        kernel_bench_dir = (curr_dir / "../../KernelBench").resolve()

        level_dir = kernel_bench_dir / f"level{self.level}"

        tasks = []

        for filename in os.listdir(level_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[0])
            
            file_path = level_dir / filename

            with open(file_path) as f:
                code = f.read()
            
            tasks.append({
                "code": code,
                "sample_id": task - 1,
                "problem_id": task
            })

        tasks.sort(key=lambda x: x["sample_id"])

        return tasks

    def good_kernels_blog_solutions(self):
        sols_dir = (curr_dir / "../../../good-kernels/solutions").resolve()

        tasks = []

        for filename in os.listdir(sols_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[1].split(".py")[0])
            
            file_path = sols_dir / filename

            with open(file_path) as f:
                code = f.read()
            
            for _ in range(self.dup_count):
                tasks.append({
                    "code": code,
                    "problem_id": task
                })

        tasks.sort(key=lambda x: x["problem_id"])

        for idx, task in enumerate(tasks):
            task["sample_id"] = idx

        return tasks


    async def eval_samples(self, samples):
        eval_id_to_sample = {}

        for sample in samples:
            eval_id = str(uuid.uuid4())
            # sample_id = sample["sample_id"]
            problem_id = sample["problem_id"]
            code = sample["code"]

            eval_id_to_sample[eval_id] = sample

            await self.disk_channel.send({
                "id": eval_id,
                "level": self.level,
                "task": problem_id,
                "code": code,
                "mode": self.mode
            })

        all_results = []

        eval_ids = list(eval_id_to_sample.keys())

        num_samples = len(samples)
        for recv_idx in range(num_samples):
            # recv, then get the sample based on the eval_id
            if self.dry_run:
                res = {
                    "id": eval_ids[recv_idx],
                    "results": {
                        "stdout": "this is dry_run stdout",
                        "stderr": "this is dry_run stderr",
                        "timed_out": False,
                        "eval_results": {
                            "loaded": True,
                            "correct": True,
                            "runtime": random.random() * 10,
                            "max_diff": [0.0001],
                        }
                    }
                }
            else:
                res = await self.disk_channel.recv()
            eval_id = res["id"]
            results = res["results"]
            sample = eval_id_to_sample[eval_id]
            sample_id = sample["sample_id"]
            problem_id = sample["problem_id"]

            print(f"Received eval result for sample: {sample_id}")

            results_data = {
                "sample_id": sample_id,
                "problem_id": problem_id,
                "results": results,
            }

            all_results.append(results_data)
        
        all_results.sort(key=lambda x: x["sample_id"])

        return all_results

    async def run(self):
        if self.solutions_name == "baseline":
            samples = self.ground_truth_solutions()
        elif self.solutions_name == "metr":
            samples = self.metr_solutions()
        elif self.solutions_name == "good_kernels":
            samples = self.good_kernels_blog_solutions()
        else:
            raise Exception(f"Unexpected solutions name value: {self.solutions_name}")

        results = await self.eval_samples(samples)

        results_path = self.results_dir / f"{self.run_name}.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int)
    parser.add_argument("--solutions", type=str, default="baseline")
    parser.add_argument("--mode", type=str, default="eager")
    parser.add_argument("--run_name", type=str, required=False)
    parser.add_argument("--run_dir", type=str, required=False)
    parser.add_argument("--worker_input_dir", type=str, required=False)
    parser.add_argument("--worker_output_dir", type=str, required=False)
    parser.add_argument("--dry_run", action='store_true')
    args = parser.parse_args()

    valid_modes = [
        "eager",
        "compile"
    ]

    mode = args.mode
    if mode not in valid_modes:
        raise Exception(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")

    valid_solutions_names = [
        "baseline",
        "metr",
        "good_kernels" # (Stanford blog post with preliminary good kernels)
    ]

    solutions_name = args.solutions
    if solutions_name not in valid_solutions_names:
        raise Exception(f"Invalid solutions value: {solutions_name}. Valid solutions values are: {valid_solutions_names}")

    results_dir = (curr_dir / "../../results/eval_solutions" / solutions_name / mode).resolve()
    if args.run_dir is not None:
        results_dir = Path(args.run_dir)

    os.makedirs(results_dir, exist_ok=True)

    worker_input_dir = (curr_dir / "../../worker_io/input").resolve()

    if args.worker_input_dir is not None:
        worker_input_dir = Path(args.worker_input_dir)

    worker_output_dir = (curr_dir / "../../worker_io/output").resolve()

    if args.worker_output_dir is not None:
        worker_output_dir = Path(args.worker_output_dir)

    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.run_name is not None:
        run_name = args.run_name

    eval_sol = EvalSolutions(args.level, mode, solutions_name, run_name, results_dir, worker_input_dir, worker_output_dir, args.dry_run)
    await eval_sol.run()

if __name__ == "__main__":
    asyncio.run(main())
