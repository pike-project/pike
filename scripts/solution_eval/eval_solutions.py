import random
import logging
import os, sys
from pathlib import Path
import asyncio
import subprocess
import uuid
import json
import argparse
from datetime import datetime

logger = logging.getLogger("eval_solutions")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
root_dir = (curr_dir / "../..").resolve()
deps_dir = root_dir / "local/deps"

from src.util.disk_channel_client import DiskChannelClient

class EvalSolutions:
    def __init__(self, level: int, mode: str, solutions_name: str, title: str, run_dir: Path, output_name: str, output_dir: Path, eval_port: int, dry_run: bool, sequential: bool, verbose: bool = False):
        self.eval_port = eval_port
        self.base_url = f"http://localhost:{eval_port}"

        self.run_dir = run_dir
        if not dry_run:
            self.eval_client = DiskChannelClient(port=eval_port)
        self.output_dir = output_dir

        self.level = level
        self.mode = mode
        self.title = title
        self.solutions_name = solutions_name
        self.output_name = output_name
        self.dry_run = dry_run
        self.sequential = sequential

        if verbose:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
                logger.addHandler(handler)

    def dry_run_metr_samples(self):
        """Generate placeholder samples using ground-truth task IDs (no METR repo needed)."""
        gt_samples = self.ground_truth_solutions()
        return [{"code": "# dry-run placeholder", "problem_id": s["problem_id"]} for s in gt_samples]

    def metr_solutions(self):
        kernel_bench_dir = (deps_dir / "KernelBenchFiltered").resolve()

        level = self.level
        if level == "3-metr":
            level = "3"

        level_dir = kernel_bench_dir / f"best_agent_solutions/level_{level}"

        tasks = []

        for filename in os.listdir(level_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[1].split(".py")[0])

            file_path = level_dir / filename

            with open(file_path) as f:
                code = f.read()

            tasks.append({
                "code": code,
                "problem_id": task
            })

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
                "problem_id": task
            })

        return tasks

    def good_kernels_blog_solutions(self):
        sols_dir = (deps_dir / "good-kernels/solutions").resolve()

        tasks = []

        for filename in os.listdir(sols_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[1].split(".py")[0])

            file_path = sols_dir / filename

            with open(file_path) as f:
                code = f.read()

            tasks.append({
                "code": code,
                "problem_id": task
            })

        return tasks

    def agent_solutions(self):
        sols_dir = (self.run_dir / "best_solutions").resolve()

        tasks = []

        for filename in os.listdir(sols_dir):
            if not filename.endswith(".py"):
                continue

            task = int(filename.split("_")[-1].split(".py")[0])
            file_path = sols_dir / filename

            with open(file_path) as f:
                code = f.read()

            tasks.append({
                "code": code,
                "problem_id": task
            })

        return tasks

    def get_samples(self):
        if self.solutions_name == "baseline":
            samples = self.ground_truth_solutions()
        elif self.solutions_name == "agent":
            samples = self.agent_solutions()
        elif self.solutions_name == "metr":
            if self.dry_run:
                samples = self.dry_run_metr_samples()
            else:
                samples = self.metr_solutions()
        elif self.solutions_name == "good_kernels":
            samples = self.good_kernels_blog_solutions()
        else:
            raise Exception(f"Unexpected solutions name value: {self.solutions_name}")

        samples.sort(key=lambda x: x["problem_id"])

        for idx, task in enumerate(samples):
            task["sample_id"] = idx

        return samples

    async def wait_for_ready(self):
        if self.dry_run:
            return
        print("Waiting for eval server...")
        await asyncio.to_thread(self.eval_client.wait_for_ready)
        print("Eval server ready.")

    def _make_dry_run_result(self, eval_id):
        return {
            "id": eval_id,
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
            },
        }

    def _process_result(self, data, sample):
        """Extract results_data from a poll response. Always returns a dict (never raises)."""
        sample_id = sample["sample_id"]
        problem_id = sample["problem_id"]

        try:
            results = data.get("results", {})
        except Exception:
            results = {}

        runtime = None
        try:
            runtime = results["eval_results"]["runtime"]
        except Exception:
            print(f"================================================================")
            print(f"------------------- Task {problem_id} stdout -------------------")
            print(results.get("stdout"))
            print(f"------------------- Task {problem_id} stderr -------------------")
            print(results.get("stderr"))
            print(f"================================================================")

        print(f"Eval result for task: {problem_id}, runtime: {runtime}")

        return {
            "sample_id": sample_id,
            "problem_id": problem_id,
            "results": results,
            "runtime": runtime,
        }

    async def eval_samples(self, samples):
        eval_id_to_sample = {}

        for sample in samples:
            problem_id = sample["problem_id"]
            code = sample["code"]

            if self.dry_run:
                eval_id = str(uuid.uuid4())
            else:
                logger.debug("submitting task=%d code_len=%d", problem_id, len(code))
                try:
                    eval_id = self.eval_client.submit(code=code, level=self.level, task=problem_id, mode=self.mode)
                    logger.debug("submitted task=%d eval_id=%s", problem_id, eval_id)
                except Exception as e:
                    print(f"Submit error for task {problem_id}: {e}")
                    continue

            eval_id_to_sample[eval_id] = sample

        all_results = []

        for eval_id, sample in eval_id_to_sample.items():
            problem_id = sample["problem_id"]
            logger.debug("polling for eval_id=%s task=%d", eval_id, problem_id)
            if self.dry_run:
                data = self._make_dry_run_result(eval_id)
            else:
                data = await asyncio.to_thread(self.eval_client.poll_for_result, eval_id)

            if data is None:
                continue

            logger.debug("poll returned for eval_id=%s task=%d", eval_id, problem_id)
            all_results.append(self._process_result(data, sample))

        all_results.sort(key=lambda x: x["sample_id"])

        return all_results

    async def eval_samples_seq(self, samples):
        all_results = []

        for sample in samples:
            problem_id = sample["problem_id"]
            code = sample["code"]
            eval_id = str(uuid.uuid4())

            if self.dry_run:
                data = self._make_dry_run_result(eval_id)
            else:
                logger.debug("submitting task=%d code_len=%d", problem_id, len(code))
                try:
                    eval_id = self.eval_client.submit(code=code, level=self.level, task=problem_id, mode=self.mode)
                except Exception as e:
                    print(f"Submit error for task {problem_id}: {e}")
                    continue

                logger.debug("submitted task=%d eval_id=%s", problem_id, eval_id)

                data = await asyncio.to_thread(self.eval_client.poll_for_result, eval_id)

            if data is None:
                continue

            all_results.append(self._process_result(data, sample))

        all_results.sort(key=lambda x: x["sample_id"])

        return all_results

    async def run(self):
        await self.wait_for_ready()

        samples = self.get_samples()

        if self.sequential:
            results = await self.eval_samples_seq(samples)
        else:
            results = await self.eval_samples(samples)

        results_path = self.output_dir / f"{self.output_name}.json"

        full_results = {
            "title": self.title,
            "results": results
        }

        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=4)

    async def close(self):
        if not self.dry_run:
            print("Sending close message to worker...")
            self.eval_client.close()


async def main():
    parser = argparse.ArgumentParser()
    # NOTE: this level must be a string, since we may be passing in level names like "3-metr" to distinguish from
    # the original KernelBench level 3
    parser.add_argument("--level", type=str)
    parser.add_argument("--solutions", type=str, default="baseline")
    parser.add_argument("--mode", type=str, default="eager")
    parser.add_argument("--run-dir", type=str, required=False)
    parser.add_argument("--output-name", type=str, required=False)
    parser.add_argument("--output-dir", type=str, required=False)
    parser.add_argument("--eval-port", type=int, default=8000)
    parser.add_argument("--dry-run", action='store_true')
    parser.add_argument("--close-worker", action='store_true')
    parser.add_argument("--sequential", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    valid_modes = [
        "eager",
        "compile",
        "tensorrt"
    ]

    valid_modes_title_map = {
        "eager": "Eager",
        "compile": "torch.compile",
        "tensorrt": "TensorRT"
    }

    mode = args.mode
    if mode not in valid_modes:
        raise Exception(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")

    valid_solutions_names = [
        "baseline",
        "agent",
        "metr",
        "good_kernels" # (Stanford blog post with preliminary good kernels)
    ]

    solutions_name_title_map = {
        "baseline": "",
        "agent": "Ours",
        "metr": "METR",
        "good_kernels": "Stanford Blog",
    }

    solutions_name = args.solutions
    if solutions_name not in valid_solutions_names:
        raise Exception(f"Invalid solutions value: {solutions_name}. Valid solutions values are: {valid_solutions_names}")

    title = solutions_name_title_map[solutions_name]

    if solutions_name == "baseline":
        title = valid_modes_title_map[mode]

    run_dir = None
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)

    if solutions_name == "agent" and run_dir is None:
        raise Exception("Evaluating agent solutions requires passing in a --run-dir argument to show where the agent run was")

    if solutions_name == "metr" or solutions_name == "good_kernels":
        subprocess.run(["bash", str(root_dir / "tools/fetch_eval_deps.sh")], check=True, cwd=root_dir)

    output_dir = (curr_dir / "../../results/eval_solutions" / solutions_name / mode).resolve()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    output_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.output_name is not None:
        output_name = args.output_name

    eval_sol = EvalSolutions(args.level, mode, solutions_name, title, run_dir, output_name, output_dir, args.eval_port, args.dry_run, args.sequential, verbose=args.verbose)
    await eval_sol.run()

    if args.close_worker:
        await eval_sol.close()

if __name__ == "__main__":
    asyncio.run(main())
