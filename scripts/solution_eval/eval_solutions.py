import os, sys
from pathlib import Path
import asyncio
import uuid
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.util.disk_channel import DiskChannel

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

class EvalSolutions:
    def __init__(self, level: int, mode: str, results_dir: Path, worker_input_dir: Path, worker_output_dir: Path):
        tx_dir = worker_input_dir
        rx_dir = worker_output_dir

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.results_dir = results_dir

        self.dup_count = 1

        self.level = level
        self.mode = mode
    
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

        num_samples = len(samples)
        for _ in range(num_samples):
            # recv, then get the sample based on the eval_id
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
        samples = self.ground_truth_solutions()
        # samples = self.metr_solutions()
        # samples = self.good_kernels_blog_solutions()

        results = await self.eval_samples(samples)

        results_path = self.results_dir / f"baseline_{self.mode}.json"
        # results_path = self.results_dir / f"good_kernels_src_2.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int)
    parser.add_argument("--mode", type=str, default="eager")
    parser.add_argument("--run_dir", type=str, required=False)
    parser.add_argument("--worker_input_dir", type=str, required=False)
    parser.add_argument("--worker_output_dir", type=str, required=False)
    args = parser.parse_args()

    results_dir = (curr_dir / "../../results/eval_solutions").resolve()
    if args.run_dir is not None:
        results_dir = Path(args.run_dir)

    os.makedirs(results_dir, exist_ok=True)

    worker_input_dir = (curr_dir / "../../worker_io/input").resolve()

    if args.worker_input_dir is not None:
        worker_input_dir = Path(worker_input_dir)

    worker_output_dir = (curr_dir / "../../worker_io/input").resolve()

    if args.worker_output_dir is not None:
        worker_output_dir = Path(worker_output_dir)

    eval_sol = EvalSolutions(args.level, args.mode, results_dir, worker_input_dir, worker_output_dir)
    await eval_sol.run()

if __name__ == "__main__":
    asyncio.run(main())
