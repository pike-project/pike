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
import src.prompt_constructor as prompt
from src.utils import extract_first_code, set_gpu_arch, read_file, create_inference_server_from_presets, maybe_multithread, maybe_multithread_ordered

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

def get_sample_dir(step_dir: Path, level, problem_id, sample_id):
    return step_dir / f"level_{level}_problem_{problem_id}_sample_{sample_id}"

def query_llm(query: str, inference_server: callable):
    try:
        raw_llm_output = inference_server(query)
        return raw_llm_output
    except Exception as e:
        print(f"Error generating sample: {e}")
        return None

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

        # num_problems_in_level = len(curr_level_dataset)

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
        self.run_dir = run_dir

        pydra.save_yaml(config.to_dict(), run_dir / "generation_config.yaml")

        self.curr_step = 0

        assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

        self.problem_id_range = problem_id_range

        tx_dir = Path(config.worker_input_dir)
        rx_dir = Path(config.worker_output_dir)

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        # Create inference function with config parameters
        # We provide some presets in utils but you can also pass in your own, see query_server for more details
        self.inference_server = create_inference_server_from_presets(server_type=self.config.server_type,
                                                                model_name=self.config.model_name,
                                                                temperature=self.config.temperature,
                                                                max_tokens=self.config.max_tokens,
                                                                verbose=self.config.verbose)

    def get_problem_code(self):
        dataset = self.curr_level_dataset
        problem_id = self.config.task

        # Fetch problem source code
        if self.config.dataset_src == "huggingface":
            curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, desc=None)

            ref_arch_src = curr_problem_row["code"][0]
            problem_name = curr_problem_row["name"][0]

        elif self.config.dataset_src == "local":
            problem_idx_in_dataset = problem_id - 1 # due to dataset list being 0-indexed locally
            ref_arch_path = dataset[problem_idx_in_dataset]

            problem_name = os.path.basename(ref_arch_path)
            ref_arch_src = read_file(ref_arch_path)

        return ref_arch_src

    def query_llm_parallel(self, queries):
        res = maybe_multithread_ordered(
            query_llm, 
            queries,
            self.config.num_workers, 
            time_interval=self.config.api_query_interval, 
            # extra args
            inference_server=self.inference_server
        )

        return res

    def get_step_dir(self):
        step_dir = self.run_dir / f"step_{self.curr_step}"
        return step_dir

    def get_sample_dir(self, sample_id):
        level = self.config.level
        problem_id = self.config.task

        sample_dir = self.get_step_dir() / f"level_{level}_problem_{problem_id}_sample_{sample_id}"
        os.makedirs(sample_dir, exist_ok=True)

        return sample_dir

    def write_sample_data(self, sample_id, filename, data):
        file_path = self.get_sample_dir(sample_id) / filename
        with open(file_path, "w") as f:
            f.write(data)

    def query_and_save(self, queries):
        for sample_id, query in enumerate(queries):
            self.write_sample_data(sample_id, "prompt.md", query)

        query_results = self.query_llm_parallel(queries)

        # important: this assumes results arrive back in the order they were sent
        for sample_id, query_result in enumerate(query_results):
            self.write_sample_data(sample_id, "query_result.md", query_result)
        
        return query_results
    
    def run_step(self, queries, extract_code=True):
        problem_id = self.config.task
        query_results = self.query_and_save(queries)

        res = []
        if extract_code:
            for sample_id, query_result in enumerate(query_results):
                custom_kernel = extract_first_code(query_result, ["python", "cpp"])
                # check LLM is able to generate custom CUDA code
                if custom_kernel is not None:
                    self.write_sample_data(sample_id, "kernel.py", custom_kernel)
                
                    res.append({
                        "sample_id": sample_id,
                        "problem_id": problem_id,
                        "code": custom_kernel,
                    })
        else:
            res = query_results

        return res

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

            self.write_sample_data(sample_id, "eval_results.json", json.dumps(results))

            results_data = {
                "sample_id": sample_id,
                "problem_id": problem_id,
                "results": results,
            }

            all_results.append(results_data)
        
        all_results.sort(key=lambda x: x["sample_id"])

        return all_results

    def run_eval(self, samples):
        all_results = asyncio.run(self.eval_samples(samples))

        working_kernel_samples = []
        correctness_fails_samples = []
        error_samples = []

        for results_data in all_results:
            sample_id = results_data["sample_id"]
            results = results_data["results"]

            # TODO: should we be using sample_id to index here?
            code = samples[sample_id]["code"]

            stdout = results["stdout"]
            stderr = results["stderr"]

            model_loaded = True

            if "eval_results" in results:
                eval_results = results["eval_results"]

                if "runtime" in eval_results:
                    runtime = eval_results["runtime"]
                    print(f"Sample {sample_id} runtime: {runtime}")

                    max_diff = eval_results["max_diff"]

                    working_kernel_samples.append({
                        "code": code,
                        "runtime": runtime,
                        "max_diff": max_diff
                    })
                elif "correct" in eval_results:
                    correct = eval_results["correct"]
                    max_diff = eval_results["max_diff"]
                    print(f"Sample {sample_id} correct: {correct}, max_diff: {max_diff}")

                    if not correct:
                        # TODO: more info would be useful here
                        correctness_fails_samples.append({
                            "code": code,
                            "max_diff": max_diff
                        })
                else:
                    model_loaded = False
            else:
                model_loaded = False

            if not model_loaded:
                error_samples.append({
                    "code": code,
                    "stdout": stdout,
                    "stderr": stderr
                })

            # TODO: it may be useful to pass stdout back to the LLM even if the run was successful,
            # in case the LLM wants to test something by printing, etc.

            print(f"\n----------- Sample {sample_id} stdout ------------")
            print(stdout)
            print(f"----------- Sample {sample_id} stderr ------------")
            print(stderr)
            print("------------------------------------------------\n")

        self.curr_step += 1

        return (working_kernel_samples, correctness_fails_samples, error_samples)

    # TODO: this should save working solutions to our "solutions database"
    def save_working_solutions(self, eval_data):
        (working_kernel_samples, _, _) = eval_data
        pass

    # returns the prompts to make in the next error/correctness-fixing step, if any are needed
    # (if no error/correctness-fixing is needed, simply proceed to the next code gen step)
    def get_correction_queries_and_bad_solutions(self, eval_data):
        (_, correctness_fails_samples, error_samples) = eval_data

        queries = []
        bad_solutions = []

        # TODO
        for sample in correctness_fails_samples:
            pass

        for sample in error_samples:
            code = sample["code"]
            stdout = sample["stdout"]
            stderr = sample["stderr"]
            queries.append(prompt.prompt_summarize_error(code, stdout, stderr))
            bad_solutions.append(code)

        return queries, bad_solutions

    # in this case, we are creating queries that will prompt the LLM to fix the
    # error directly from stdout and stderr, no error summarizing involved
    def get_direct_fix_queries(self, eval_data):
        (_, correctness_fails_samples, error_samples) = eval_data

        queries = []

        # TODO
        for sample in correctness_fails_samples:
            pass

        problem_code = self.get_problem_code()

        for sample in error_samples:
            code = sample["code"]
            stdout = sample["stdout"]
            stderr = sample["stderr"]
            queries.append(prompt.prompt_fix_compile_stdout_stderr(problem_code, code, stdout, stderr))

        return queries

    # this is an example of the first round of querying the LLM, before there is any prior output
    def run_init_queries(self, num_queries):
        problem_code = self.get_problem_code()

        custom_cuda_prompt = prompt.prompt_generate_custom_cuda_from_prompt_template(problem_code)

        queries = []

        for _ in range(num_queries):
            queries.append(custom_cuda_prompt)

        res = self.query_llm_parallel(queries)

        return res

    # TODO: the initial queries are currently all the same, we should vary them to diversify the initial set of solutions
    def get_initial_queries(self):
        problem_code = self.get_problem_code()

        queries = []

        for sample_id in range(self.config.num_samples):
            custom_cuda_prompt = prompt.prompt_generate_custom_cuda_from_prompt_template(problem_code)
            queries.append(custom_cuda_prompt)
        
        return queries

    def gen_samples(self, queries=None):
        res = self.run_step(queries, extract_code=True)
        return res

    # in this case, we are having a "summary agent" summarize the errors, not
    # attempt to fix the errors
    def gen_summarized_fix_messages(self, eval_data):
        correction_queries, bad_solutions = self.get_correction_queries_and_bad_solutions(eval_data)
        res = self.run_step(correction_queries, extract_code=False)

        # craft a message which will prompt the LLM to fix the broken solution in the next step
        problem_code = self.get_problem_code()
        queries = []
        for error_summary, solution_to_fix in zip(res, bad_solutions):
            if error_summary is None:
                continue

            prompt = prompt.prompt_fix_compile_summarized(problem_code, solution_to_fix, error_summary)
            queries.append(prompt)

        self.curr_step += 1

        return queries

    def run(self):
        # initial_queries = self.get_initial_queries()
        # samples = self.gen_samples(initial_queries)
        # eval_data = self.run_eval(samples)
        # self.save_working_solutions(eval_data)

        # error_summary_queries = self.gen_error_summary_messages(eval_data)
        # new_samples = self.gen_samples(error_summary_queries)
        # new_eval_data = self.run_eval(new_samples)
        # self.save_working_solutions(new_eval_data)

        queries = self.get_initial_queries()

        for i in range(5):
            print(f"======================= Running fix iteration {i} =======================")
            new_samples = self.gen_samples(queries)
            eval_data = self.run_eval(new_samples)
            self.save_working_solutions(eval_data)
            queries = self.get_direct_fix_queries(eval_data)
            if len(queries) == 0:
                print(f"======== All solutions passing correctness, exiting at iteration {i} ========")
                break


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

