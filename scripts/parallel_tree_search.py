import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import asyncio
import uuid


from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch, read_file, create_inference_server_from_presets, maybe_multithread

from src.util.disk_channel import DiskChannel

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

REPO_TOP_PATH = Path.resolve(curr_dir / "..")
KERNEL_BENCH_PATH = REPO_TOP_PATH / "KernelBench"

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class GenerationConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

        self.task = REQUIRED
        
        # subset of problems to generate, otherwise generate on all problems in the level
        # both sides are inclusive
        # (None, None) -> full range
        # self.subset = (self.task, self.task) # range of problems to generate samples for

        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # self.run_name = REQUIRED # name of the run
        self.run_name = timestamp_str

        # num of thread pool to call inference server in parallel
        self.num_workers = 1
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.8
        
        # Logging

        # absolute path to data dir
        self.data_dir = REQUIRED
        # Top Directory to Store Runs
        # self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # these are for the eval worker
        self.worker_input_dir = REQUIRED
        self.worker_output_dir = REQUIRED
    
        self.verbose = False
        self.store_type = "local" # TODO: add Database Integration

        # Future support
        # Migrate Monkeys code base to KernelBench
        self.num_samples = REQUIRED # for sampling multiple samples per problem

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"
    

@dataclass
class WorkArgs:
    problem_id: int # logically indexed
    sample_id: int

def get_sample_dir(layer_dir: Path, level, problem_id, sample_id):
    return layer_dir / f"level_{level}_problem_{problem_id}_sample_{sample_id}"

def generate_sample_single(work: WorkArgs, config: GenerationConfig, dataset, inference_server: callable, layer_dir: str) -> bool:
    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == work.problem_id, desc=None)

        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = work.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == work.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

    sample_dir = get_sample_dir(layer_dir, config.level, work.problem_id, work.sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # Construct Prompt   
    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)

    prompt_path = sample_dir / "prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    raw_llm_output = inference_server(custom_cuda_prompt)

    raw_llm_output_path = sample_dir / "raw_llm_output.txt"
    with open(raw_llm_output_path, "w") as f:
        f.write(raw_llm_output)

    custom_cuda = extract_first_code(raw_llm_output, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(f"Generated sample {work.sample_id} for problem {problem_number}: {problem_name}")

    # Store to local file
    kernel_path = sample_dir / "kernel.py"
    with open(kernel_path, "w") as f:
        f.write(custom_cuda)
    
    return {
        "problem_id": work.problem_id,
        "sample_id": work.sample_id,
        "code": custom_cuda
    }
    

def generate_sample_launcher(work: WorkArgs, config: GenerationConfig, dataset, inference_server: callable, layer_dir: str):
    try:
        return generate_sample_single(work, config, dataset, inference_server, layer_dir)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


def check_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    return os.path.exists(kernel_path)

class ParallelTreeSearch:
    def __init__(self, config):
        self.config = config

        # Dataset Configurations
        if config.dataset_src == "huggingface":
            dataset = load_dataset(config.dataset_name)
            curr_level_dataset = dataset[f"level_{config.level}"]
        elif config.dataset_src == "local":
            curr_level_dataset = construct_kernelbench_dataset(KERNEL_BENCH_PATH, config.level)

        self.curr_level_dataset = curr_level_dataset

        num_problems_in_level = len(curr_level_dataset)

        problem_id_range = range(config.task, config.task + 1)
        # if config.subset == (None, None):
        #     problem_id_range = range(1, num_problems_in_level)
        # else:
        #     # assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        #     problem_id_range = range(config.subset[0], config.subset[1])

        print(f"Generating {config.num_samples} samples each for level {config.level} problems: {problem_id_range}")

        data_dir = Path(config.data_dir)

        runs_dir = data_dir / "runs"

        # set up run directory
        run_dir = runs_dir / config.run_name
        os.makedirs(run_dir, exist_ok=True)
        pydra.save_yaml(config.to_dict(), run_dir / "generation_config.yaml")

        layer_dir = run_dir / "layer_0"
        os.makedirs(layer_dir, exist_ok=True)

        self.layer_dir = layer_dir

        assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

        self.problem_id_range = problem_id_range

        tx_dir = Path(config.worker_input_dir)
        rx_dir = Path(config.worker_output_dir)

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

    def generate_samples(self):
        problems_to_run = []
        for problem_id in self.problem_id_range: # end index is inclusive
            # assume sample id is 0 for now
            for sample_id in range(self.config.num_samples):
                problems_to_run.append(
                    WorkArgs(
                        problem_id=int(problem_id),
                        sample_id=sample_id
                    )
                )

        self.problems_to_run = problems_to_run

        # Create inference function with config parameters
        # We provide some presets in utils but you can also pass in your own, see query_server for more details
        inference_server = create_inference_server_from_presets(server_type=self.config.server_type,
                                                            model_name=self.config.model_name,
                                                            temperature=self.config.temperature,
                                                            max_tokens=self.config.max_tokens,
                                                            verbose=self.config.verbose)

        # Launch workers
        generation_results = maybe_multithread(generate_sample_launcher, 
                        self.problems_to_run, 
                        self.config.num_workers, 
                        time_interval=self.config.api_query_interval, 
                        # extra args
                        config=self.config, 
                        dataset=self.curr_level_dataset, 
                        inference_server=inference_server,
                        layer_dir=self.layer_dir
                        )

        num_generated_samples = len(generation_results)
        total_problems = len(problems_to_run)
        num_failed_problems = total_problems - num_generated_samples
        print(f"Generated {num_generated_samples} samples for total {total_problems} problems, Please retry for the {num_failed_problems} failed problems.")

        return generation_results

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
                "level": self.config.level,
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

            sample_dir = get_sample_dir(self.layer_dir, self.config.level, sample["problem_id"], sample["sample_id"])
            os.makedirs(sample_dir, exist_ok=True)

            results_path = sample_dir / "eval_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f)

            results_data = {
                "sample_id": sample_id,
                "problem_id": problem_id,
                "results": results,
            }

            all_results.append(results_data)
        
        all_results.sort(key=lambda x: x["sample_id"])

        return all_results

    def eval_and_process(self, samples):
        all_results = asyncio.run(self.eval_samples(samples))

        for results_data in all_results:
            sample_id = results_data["sample_id"]
            results = results_data["results"]

            if "eval_results" in results:
                eval_results = results["eval_results"]

                if "runtime" in eval_results:
                    runtime = eval_results["runtime"]
                    print(f"Sample {sample_id} runtime: {runtime}")
                elif "correct" in eval_results:
                    correct = eval_results["correct"]
                    max_diff = eval_results["max_diff"]
                    print(f"Sample {sample_id} correct: {correct}, max_diff: {max_diff}")
            
            stderr = results["stderr"]
            if stderr is not None:
                print(f"\n------- Sample {sample_id} has stderr --------")
                print(stderr)
                print("------------------------------------------\n")

    def run(self):
        samples = self.generate_samples()

        self.eval_and_process(samples)


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    print(f"Starting Batch Generation with config: {config}")

    tree_search = ParallelTreeSearch(config)
    tree_search.run()


if __name__ == "__main__":
    main()

