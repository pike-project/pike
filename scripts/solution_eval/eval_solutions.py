import os, sys
from pathlib import Path
import asyncio
import uuid
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.util.disk_channel import DiskChannel

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

class EvalSolutions:
    def __init__(self):
        tx_dir = (curr_dir / "../../worker_io/input").resolve()
        rx_dir = (curr_dir / "../../worker_io/output").resolve()

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.results_dir = (curr_dir / "../../results/eval_solutions").resolve()
        os.makedirs(self.results_dir, exist_ok=True)

        self.level = 3
    
    def get_ground_truth_solutions(self):
        kernel_bench_dir = (curr_dir / "../../KernelBench").resolve()

        level_dir = kernel_bench_dir / f"level{self.level}"

        tasks = []

        sample_id = 0

        for filename in os.listdir(level_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[0])
            
            file_path = level_dir / filename

            with open(file_path) as f:
                code = f.read()
            
            tasks.append({
                "code": code,
                "sample_id": sample_id,
                "problem_id": task
            })

            sample_id += 1

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
                "code": code
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
        samples = self.get_ground_truth_solutions()
        results = await self.eval_samples(samples)

        results_path = self.results_dir / "out.json"

        with open(results_path) as f:
            json.dump(results, f, indent=4)


async def main():
    eval_sol = EvalSolutions()
    await eval_sol.run()

if __name__ == "__main__":
    asyncio.run(main())
