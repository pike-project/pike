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
from enum import Enum


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

class EvalState(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ERROR = "error"

@dataclass
class Query:
    sample_id: int
    query: str

@dataclass
class QueryResult:
    sample_id: int
    result: str

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
        # Max spend of $0.20 for a solution output with current Gemini pricing:
        # $10.00 per 1M tokens
        # 20000 * 10 / (10^6) = 0.20
        self.max_tokens = 20000
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

    def query_llm_parallel(self, raw_queries: list[str]) -> list[str]:
        res = maybe_multithread_ordered(
            query_llm, 
            raw_queries,
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

    def query_and_save(self, queries: list[Query]):
        raw_queries = []
        sample_ids = []

        for q in queries:
            self.write_sample_data(q.sample_id, "prompt.md", q.query)
            raw_queries.append(q.query)
            sample_ids.append(q.sample_id)

        query_results = self.query_llm_parallel(raw_queries)

        filtered_query_results = []

        # important: this assumes results arrive back in the order they were sent
        for sample_id, query_result in zip(sample_ids, query_results):
            if query_result is None:
                print(f"No query result for sample: {sample_id}")
                continue

            self.write_sample_data(sample_id, "query_result.md", query_result)
            filtered_query_results.append(QueryResult(sample_id=sample_id, result=query_result))
        
        return filtered_query_results

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

        final_results = []

        correct_count = 0
        incorrect_count = 0
        error_count = 0

        for results_data in all_results:
            sample_id = results_data["sample_id"]
            results = results_data["results"]

            code = None
            for sample in samples:
                if sample["sample_id"] == sample_id:
                    code = sample["code"]
            
            assert code is not None, "sample_id code was not found in original samples"

            stdout = results["stdout"]
            stderr = results["stderr"]
            timed_out = results["timed_out"]

            # sample states are: "correct", "incorrect", "error"
            sample_data = {
                "state": EvalState.ERROR,
                "code": code,
                "sample_id": sample_id
            }

            model_loaded = True

            if "eval_results" in results:
                eval_results = results["eval_results"]

                if "runtime" in eval_results:
                    runtime = eval_results["runtime"]
                    print(f"Sample {sample_id} runtime: {runtime}")

                    max_diff = eval_results["max_diff"]

                    sample_data["state"] = EvalState.CORRECT
                    sample_data["runtime"] = runtime
                    sample_data["max_diff"] = max_diff
                elif "correct" in eval_results:
                    correct = eval_results["correct"]
                    max_diff = eval_results["max_diff"]
                    print(f"Sample {sample_id} correct: {correct}, max_diff: {max_diff}")

                    assert not correct, "If there is no runtime for a given sample, we expect it to be not correct"

                    if not correct:
                        # TODO: more info would be useful here

                        sample_data["state"] = EvalState.INCORRECT
                        sample_data["max_diff"] = max_diff
                else:
                    model_loaded = False
            else:
                model_loaded = False

            if not model_loaded:
                sample_data["state"] = EvalState.ERROR
                sample_data["results"] = results

            # TODO: it may be useful to pass stdout back to the LLM even if the run was successful,
            # in case the LLM wants to test something by printing, etc.

            # print(f"\n----------- Sample {sample_id} stdout ------------")
            # print(stdout)
            # print(f"----------- Sample {sample_id} stderr ------------")
            # print(stderr)
            # print(f"----------- Sample {sample_id} timed_out ------------")
            # print(timed_out)
            # print("------------------------------------------------\n")

            final_state = sample_data["state"]
            if final_state == EvalState.CORRECT:
                correct_count += 1
            elif final_state == EvalState.INCORRECT:
                incorrect_count += 1
            elif final_state == EvalState.ERROR:
                error_count += 1

            final_results.append(sample_data)

        print(f"\n-------------- Step: {self.curr_step} --------------\n")
        print(f"CORRECT: {correct_count}, INCORRECT: {incorrect_count}, ERROR: {error_count}")
        print("------------------------------------------------\n")

        self.curr_step += 1

        return final_results

    # TODO: this should save working solutions to our "solutions database"
    def save_working_solutions(self, eval_data):
        for sample_data in eval_data:
            if sample_data["state"] != EvalState.CORRECT:
                continue

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
    def get_direct_fix_queries(self, eval_data) -> list[Query]:
        queries = []

        problem_code = self.get_problem_code()

        for sample_data in eval_data:
            sample_id = sample_data["sample_id"]
            state = sample_data["state"]

            if state == EvalState.INCORRECT:
                code = sample_data["code"]
                max_diff = sample_data["max_diff"]
                query = prompt.prompt_fix_correctness(problem_code, code, max_diff)
                queries.append(Query(sample_id=sample_id, query=query))
            elif state == EvalState.ERROR:
                code = sample_data["code"]
                results = sample_data["results"]
                query = prompt.prompt_fix_compile_stdout_stderr(problem_code, code, results)
                queries.append(Query(sample_id=sample_id, query=query))

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
    def get_init_queries(self) -> list[Query]:
        problem_code = self.get_problem_code()

        queries = []

        for sample_id in range(self.config.num_samples):
            custom_cuda_prompt = prompt.prompt_generate_custom_cuda_from_prompt_template(problem_code)
            queries.append(Query(sample_id=sample_id, query=custom_cuda_prompt))
        
        return queries

    # sample_ids_and_queries is a zipped list of pairs (sample_id, query)
    def gen_samples(self, queries: list[Query]):
        query_results = self.query_and_save(queries)

        problem_id = self.config.task

        res = []
        for qr in query_results:
            custom_kernel = extract_first_code(qr.result, ["python", "cpp"])
            # check LLM is able to generate custom CUDA code
            if custom_kernel is None:
                print(f"Failed to parse custom kernel for sample: {qr.sample_id}")
                continue

            self.write_sample_data(qr.sample_id, "kernel.py", custom_kernel)
        
            res.append({
                "sample_id": qr.sample_id,
                "problem_id": problem_id,
                "code": custom_kernel,
            })

        return res

    # in this case, we are having a "summary agent" summarize the errors, not
    # attempt to fix the errors
    def gen_summarized_fix_messages(self, eval_data):
        correction_queries, bad_solutions = self.get_correction_queries_and_bad_solutions(eval_data)
        res = self.query_and_save(correction_queries)

        # craft a message which will prompt the LLM to fix the broken solution in the next step
        problem_code = self.get_problem_code()
        queries = []
        for error_summary, solution_to_fix in zip(res, bad_solutions):
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

        queries = self.get_init_queries()

        for i in range(3):
            print(f"======================= Running fix iteration {i} =======================")
            new_samples = self.gen_samples(queries)
            eval_data = self.run_eval(new_samples)
            self.save_working_solutions(eval_data)
            queries = self.get_direct_fix_queries(eval_data)
            if len(queries) == 0:
                print(f"======== All solutions passing/failed, exiting at iteration {i} ========")
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

