import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
import asyncio
import uuid
import random
from enum import Enum
import requests


from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
import src.prompt_constructor as prompt
import src.query_strategies as query_strategies
from src.utils import extract_first_code, extract_idea_list, set_gpu_arch, read_file, create_inference_server_from_presets, maybe_multithread, maybe_multithread_ordered

import src.util.query_util as query_util

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
    problem_id: int
    sample_id: int
    branch: int
    query: str

@dataclass
class QueryResult:
    problem_id: int
    sample_id: int
    branch: int
    result: str

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

class GenerationConfig(Config):
    def __init__(self):
        # Problem Specification
        self.level = REQUIRED

        self.task_start = REQUIRED
        self.task_end = REQUIRED
        
        # subset of problems to generate, otherwise generate on all problems in the level
        # both sides are inclusive
        # (None, None) -> full range
        # self.subset = (self.task, self.task) # range of problems to generate samples for

        self.run_dir = None

        # num of thread pool to call inference server in parallel
        self.num_workers = 1
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        # Max spend of $0.30 for a solution output with current Gemini pricing:
        # $10.00 per 1M tokens
        # 30000 * 10 / (10^6) = 0.30
        self.max_tokens = 30000
        self.temperature = 0.8
        
        # Logging

        # absolute path to data dir
        self.data_dir = ""
        # Top Directory to Store Runs
        # self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # these are for the eval worker
        # TODO: these can be removed
        self.worker_input_dir = REQUIRED
        self.worker_output_dir = REQUIRED

        # TODO: maybe add dry_run_eval and dry_run_queries
        self.dry_run = False
    
        self.verbose = False
        self.store_type = "local"

        # Future support
        # Migrate Monkeys code base to KernelBench
        self.num_samples = REQUIRED # for sampling multiple samples per problem

        self.num_phases = REQUIRED
        # note: this does not include the initial attempt
        self.max_fix_attempts = REQUIRED

        self.eval_port = REQUIRED

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"
    

@dataclass
class WorkArgs:
    problem_id: int # logically indexed
    sample_id: int

def query_llm(query: str, inference_server: callable):
    retry_attempts = 5
    for _ in range(retry_attempts):
        try:
            raw_llm_output = inference_server(query)
            return raw_llm_output
        except Exception as e:
            print(f"Error generating sample: {e}")
            sleep(10)
    
    return None

class ParallelTreeSearch:
    def __init__(self, config):
        self.config = config
        
        curr_level_dataset = construct_kernelbench_dataset(KERNEL_BENCH_PATH, config.level)
        self.curr_level_dataset = curr_level_dataset

        # num_problems_in_level = len(curr_level_dataset)

        self.problem_id_to_path = {}
        self.problem_ids = []

        for problem_path in self.curr_level_dataset:
            problem_id = int(os.path.basename(problem_path).split("_")[0])

            if problem_id >= self.config.task_start and problem_id <= self.config.task_end:
                self.problem_ids.append(problem_id)
                self.problem_id_to_path[problem_id] = problem_path

        print(f"Generating {config.num_samples} samples each for level {config.level} problems: {self.problem_ids}")

        if config.run_dir is None and config.data_dir is None:
            raise Exception("Either run_dir or data_dir must be set in config")

        if config.run_dir is None:
            data_dir = Path(config.data_dir)
            run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            runs_dir = data_dir / "runs"
            run_dir = runs_dir / run_name
            self.run_dir = run_dir
        else:
            self.run_dir = Path(config.run_dir)
        
        os.makedirs(self.run_dir, exist_ok=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=4)

        self.all_solutions = {}
        self.phase_solutions_by_branch = {}
        self.phase_solutions = {}

        self.query_budget = 300
        self.budget_used = {}

        self.curr_phase = 0
        self.curr_step = 0

        assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

        for problem_id in self.problem_ids:
            self.phase_solutions[problem_id] = []
            self.all_solutions[problem_id] = []
            self.phase_solutions_by_branch[problem_id] = {}

        server_type = self.config.server_type
        model_name = self.config.model_name
        if self.config.dry_run:
            server_type = "cborg"
            model_name = "lbl/llama"

        # Create inference function with config parameters
        # We provide some presets in utils but you can also pass in your own, see query_server for more details
        self.inference_server = create_inference_server_from_presets(server_type=server_type,
                                                                model_name=model_name,
                                                                temperature=self.config.temperature,
                                                                max_tokens=self.config.max_tokens,
                                                                verbose=self.config.verbose)

    async def init(self):
        print("Starting handshake with worker...")

        if not self.config.dry_run:
            base_url = f"http://localhost:{self.config.eval_port}"

            while True:
                try:
                    res = requests.get(f"{base_url}/ready")
                    ready_str = res.text

                    if ready_str == "true":
                        break
                except Exception as e:
                    continue

        print("Worker handshake complete.")

    def inc_budget(self, problem_id, count):
        if problem_id in self.budget_used:
            if self.budget_used[problem_id] >= self.query_budget:
                return False

            self.budget_used[problem_id] += count
        else:
            self.budget_used[problem_id] = count
        
        return True

    def get_problem_code(self, problem_id):
        ref_arch_src = read_file(self.problem_id_to_path[problem_id])

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

    def get_task_dir(self, problem_id):
        level = self.config.level

        task_dir = self.run_dir / f"levels/level_{level}/task_{problem_id}"

        return task_dir

    def get_phase_dir(self, problem_id):
        phase_dir = self.get_task_dir(problem_id) / f"phases/phase_{self.curr_phase}"
        os.makedirs(phase_dir, exist_ok=True)

        return phase_dir

    def get_solutions_dir(self, solution_id, problem_id):
        solutions_dir = self.get_phase_dir(problem_id) / f"solutions/solution_{solution_id}"
        os.makedirs(solutions_dir, exist_ok=True)

        return solutions_dir

    def get_sample_dir(self, problem_id, sample_id):
        sample_dir = self.get_phase_dir(problem_id) / f"agents/agent_{sample_id}/step_{self.curr_step}"
        os.makedirs(sample_dir, exist_ok=True)

        return sample_dir

    def write_sample_data(self, problem_id, sample_id, filename, data):
        if sample_id is None:
            # this is for saving ideas that are general to the phase, not specific to an agent
            file_path = self.get_phase_dir(problem_id) / filename
        else:
            # this is for saving agent progress
            file_path = self.get_sample_dir(problem_id, sample_id) / filename
        with open(file_path, "w") as f:
            f.write(data)

    def query_and_save(self, queries: list[Query], idea_queries=False, prompt_name="prompt", result_name="query_result"):
        raw_queries = []
        sample_id_problem_id = []

        for q in queries:
            self.write_sample_data(q.problem_id, q.sample_id, f"{prompt_name}.md", q.query)
            raw_queries.append(q.query)
            sample_id_problem_id.append((q.sample_id, q.problem_id, q.branch))


        if self.config.dry_run:
            query_results = []
            for q in queries:
                if idea_queries:
                    res_text = "- idea1\n- idea2\n- idea 3\n"
                else:
                    res_text = "```exit(0)```"
                
                query_results.append((res_text, {}))
        else:
            query_results = self.query_llm_parallel(raw_queries)

        filtered_query_results = []

        # important: this assumes results arrive back in the order they were sent
        for (sample_id, problem_id, branch), res_data in zip(sample_id_problem_id, query_results):
            if res_data is None:
                print(f"No LLM response for sample: {sample_id}")
                continue

            (query_result, full_response) = res_data
            if query_result is None:
                print(f"No query result for sample: {sample_id}")
                print(full_response)
                if full_response is not None:
                    self.write_sample_data(problem_id, sample_id, f"{result_name}_error_llm_response.json", json.dumps(full_response, indent=4))

                continue

            self.write_sample_data(problem_id, sample_id, f"{result_name}.md", query_result)
            self.write_sample_data(problem_id, sample_id, f"{result_name}_full_llm_response.json", json.dumps(full_response, indent=4))
            filtered_query_results.append(QueryResult(problem_id=problem_id, sample_id=sample_id, branch=branch, result=query_result))
        
        return filtered_query_results

    async def eval_samples(self, samples):
        eval_id_to_sample = {}

        base_url = f"http://localhost:{self.config.eval_port}"

        for sample in samples:
            eval_id = str(uuid.uuid4())
            problem_id = sample["problem_id"]
            code = sample["code"]

            submit_path = "/submit"
            submit_params = {
                "code": code,
                "level": self.config.level,
                "task": problem_id
            }

            try:
                res = requests.get(f"{base_url}{submit_path}", params=submit_params)
                eval_id = res.text

                eval_id_to_sample[eval_id] = sample
            except Exception as e:
                continue

        all_results = []

        num_samples = len(samples)
        for (eval_id, sample) in eval_id_to_sample.items():

            poll_path = "/poll"
            poll_params = {"id": eval_id}

            while True:
                try:
                    res = requests.get(f"{base_url}{poll_path}", params=poll_params)
                except Exception as e:
                    asyncio.sleep(1.0)
                    continue
                data = res.json()
                if data is not None:
                    break

                asyncio.sleep(1.0)

            results = data["results"]
            sample_id = sample["sample_id"]
            problem_id = sample["problem_id"]

            # print(f"Received eval result for sample: {sample_id}")

            self.write_sample_data(problem_id, sample_id, "eval_results.json", json.dumps(results, indent=4))

            results_data = {
                "sample_id": sample_id,
                "problem_id": problem_id,
                "results": results,
            }

            all_results.append(results_data)
        
        all_results.sort(key=lambda x: x["sample_id"])

        return all_results

    def run_eval(self, samples):
        if self.config.dry_run:
            all_results = []
            for s in samples:
                if random.random() < 0.5:
                    all_results.append({
                        "sample_id": s["sample_id"],
                        "results": {
                            "stdout": "this is dry_run stdout",
                            "stderr": "this is dry_run stderr",
                            "timed_out": False,
                            "eval_results": {
                                "correct": True,
                                "runtime": random.random() * 10,
                                "max_diff": [0.0001],
                            }
                        }
                    })
                else:
                    all_results.append({
                        "sample_id": s["sample_id"],
                        "results": {
                            "stdout": "this is dry_run stdout",
                            "stderr": "this is dry_run stderr",
                            "timed_out": False,
                            "eval_results": {
                                "loaded": False,
                                # "correct": True,
                                # "runtime": random.random() * 10,
                                # "max_diff": [0.0001],
                            }
                        }
                    })
        else:
            all_results = asyncio.run(self.eval_samples(samples))

        final_results = []

        correct_count = 0
        incorrect_count = 0
        error_count = 0

        for results_data in all_results:
            sample_id = results_data["sample_id"]
            results = results_data["results"]

            code = None
            problem_id = None
            for sample in samples:
                if sample["sample_id"] == sample_id:
                    code = sample["code"]
                    problem_id = sample["problem_id"]
                    branch = sample["branch"]
            
            assert code is not None, "sample_id code was not found in original samples"

            stdout = results["stdout"]
            stderr = results["stderr"]
            timed_out = results["timed_out"]

            # sample states are: "correct", "incorrect", "error"
            sample_data = {
                "state": EvalState.ERROR,
                "code": code,
                "sample_id": sample_id,
                "problem_id": problem_id,
                "branch": branch,
                "phase": self.curr_phase,
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
                    # print(f"Sample {sample_id} correct: {correct}, max_diff: {max_diff}")

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

        print(f"\n--------------- Phase: {self.curr_phase}, Step: {self.curr_step} ---------------\n")
        print(f"CORRECT: {correct_count}, INCORRECT: {incorrect_count}, ERROR: {error_count}")
        print("------------------------------------------------\n")

        # if self.curr_phase == 0 and self.curr_step == 1 and correct_count == 0:
        #     print("\n\nSOMETHING IS LIKELY WRONG: exiting early as a safeguard\n\n")
        #     exit()

        self.curr_step += 1

        return final_results

    # this should save working solutions to our "solutions database"
    def save_solutions(self, eval_data):
        for sample_data in eval_data:
            if sample_data["state"] != EvalState.CORRECT:
                continue

            problem_id = sample_data["problem_id"]
            branch = sample_data["branch"]

            solution_id = len(self.phase_solutions[problem_id])
            solutions_dir = self.get_solutions_dir(solution_id, problem_id)

            code_path = solutions_dir / "kernel.py"
            data_path = solutions_dir / "data.json"

            with open(code_path, "w") as f:
                f.write(sample_data["code"])
            
            with open(data_path, "w") as f:
                json.dump(sample_data, f, indent=4, cls=EnumEncoder)
            
            self.phase_solutions[problem_id].append(sample_data)
            self.all_solutions[problem_id].append(sample_data)

            base_sample_sols = self.phase_solutions_by_branch[problem_id]
            if branch in base_sample_sols:
                base_sample_sols[branch].append(sample_data)
            else:
                base_sample_sols[branch] = [sample_data]

        self.sort_solutions()

    def sort_solutions(self):
        for problem_id, solutions in self.all_solutions.items():
            self.all_solutions[problem_id] = sorted(solutions, key=lambda x: x["runtime"])
        
        for solutions_by_branch in self.phase_solutions_by_branch.values():
            for branch, solutions in solutions_by_branch.items():
                solutions_by_branch[branch] = sorted(solutions, key=lambda x: x["runtime"])

    def gather_best_solutions_by_branch(self):
        best_solutions = {}

        for problem_id, solutions_by_branch in self.phase_solutions_by_branch.items():
            best_problem_solutions = []
            remaining_solutions = []

            for _, solutions in solutions_by_branch.items():
                if len(solutions) > 0:
                    best_problem_solutions.append(solutions[0])
                
                    # gather any remaining solutions, in case a particular branch had no correct solutions,
                    # in which case we pull from the best other solutions
                    for sol in solutions[1:]:
                        remaining_solutions.append(sol)

            all_sols = sorted(best_problem_solutions, key=lambda x: x["runtime"])
            all_sols += sorted(remaining_solutions, key=lambda x: x["runtime"])

            if len(all_sols) == 0:
                print(f"WARNING: all_sols length is 0 for task {problem_id}, ground truth solution will be used")

            problem_code = self.get_problem_code(problem_id)
            ground_truth_solution = {
                "state": EvalState.CORRECT,
                "code": problem_code,
                "sample_id": 0,
                "problem_id": problem_id,
                "branch": 0,
                "phase": self.curr_phase,
                "runtime": float("inf"),
                "max_diff": 0.0,
            }
            all_sols.append(ground_truth_solution)

            best_solutions[problem_id] = all_sols
        
        return best_solutions

    # returns the prompts to make in the next error/correctness-fixing step, if any are needed
    # (if no error/correctness-fixing is needed, simply proceed to the next code gen step)
    # TODO: this is broken
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

        for sample_data in eval_data:
            sample_id = sample_data["sample_id"]
            branch = sample_data["branch"]
            problem_id = sample_data["problem_id"]
            state = sample_data["state"]

            problem_code = self.get_problem_code(problem_id)

            if state == EvalState.INCORRECT:
                if not self.inc_budget(problem_id, 1):
                    continue

                code = sample_data["code"]
                max_diff = sample_data["max_diff"]
                query = prompt.prompt_fix_correctness(problem_code, code, max_diff)
                queries.append(Query(problem_id=problem_id, sample_id=sample_id, branch=branch, query=query))
            elif state == EvalState.ERROR:
                if not self.inc_budget(problem_id, 1):
                    continue

                code = sample_data["code"]
                results = sample_data["results"]
                query = prompt.prompt_fix_compile_stdout_stderr(problem_code, code, results)
                queries.append(Query(problem_id=problem_id, sample_id=sample_id, branch=branch, query=query))

        return queries

    # the initial queries were all the same before, so we vary them to diversify the initial set of solutions
    # via ideas from the brainstorming agent
    def get_init_queries(self, ideas) -> list[Query]:
        queries = []

        curr_sample_id = 0
        for problem_id in self.problem_ids:
            problem_code = self.get_problem_code(problem_id)
            problem_ideas = ideas[problem_id]
            for idea in problem_ideas:
                custom_cuda_prompt = prompt.prompt_generate_custom_cuda_from_prompt_template(problem_code, idea)
                queries.append(Query(problem_id=problem_id, sample_id=curr_sample_id, branch=curr_sample_id, query=custom_cuda_prompt))
                curr_sample_id += 1
        
        return queries

    # sample_ids_and_queries is a zipped list of pairs (sample_id, query)
    def gen_samples(self, queries: list[Query], extract_code=False):
        query_results = self.query_and_save(queries)

        res = []
        for qr in query_results:
            custom_kernel = extract_first_code(qr.result, ["python", "cpp"])
            # check LLM is able to generate custom CUDA code
            if custom_kernel is None:
                print(f"Failed to parse custom kernel for sample: {qr.sample_id}")
                continue

            self.write_sample_data(qr.problem_id, qr.sample_id, "kernel.py", custom_kernel)
        
            res.append({
                "sample_id": qr.sample_id,
                "problem_id": qr.problem_id,
                "branch": qr.branch,
                "code": custom_kernel,
            })

        return res

    # in this case, we are having a "summary agent" summarize the errors, not
    # attempt to fix the errors
    # TODO: this is currently broken
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

    # here we use all of the existing solutions for each problem, and try to divide the available num_samples in a way that
    # focuses mostly on the most promising solutions
    # we could employ a number of strategies here
    def get_next_queries(self):
        queries = []

        num_samples = self.config.num_samples

        curr_sample_id = 0

        # sols_by_problem = self.all_solutions
        sols_by_problem = self.gather_best_solutions_by_branch()

        for problem_id, solutions in sols_by_problem.items():
            if not self.inc_budget(problem_id, num_samples):
                continue

            sorted_sols = sorted(solutions, key=lambda x: x["runtime"])

            problem_code = self.get_problem_code(problem_id)
            strat_queries = query_strategies.simple_branching_strategy(sorted_sols, num_samples, problem_code, num_branches=4)

            if len(strat_queries) == 0:
                print(f"WARNING: No queries generated for task {problem_id}")

            for strat_q in strat_queries:
                queries.append(Query(problem_id=problem_id, sample_id=curr_sample_id, branch=strat_q.branch, query=strat_q.query))
                curr_sample_id += 1

        return queries

    def get_idea_queries(self):
        queries = []

        for problem_id in self.problem_ids:
            problem_code = self.get_problem_code(problem_id)
            # TODO: want to pass in a prior best solution (sampled) that we want to get ideas for
            # but we also want to ensure the diversity of the solutions
            # - should we ask the LLM to reason about which solutions are best to focus on, passing in a bunch of the prior good solutions?
            idea_prompt = prompt.prompt_generate_ideas(problem_code)
            queries.append(Query(problem_id=problem_id, sample_id=None, branch=None, query=idea_prompt))
        
        return queries

    def gen_and_extract_ideas(self, queries):
        query_results = self.query_and_save(queries, idea_queries=True, prompt_name="ideas_prompt", result_name="ideas_result")

        ideas = {}
        for qr in query_results:
            pass
            l = extract_idea_list(qr.result)

            self.write_sample_data(qr.problem_id, None, "ideas.json", json.dumps(l, indent=4))

            # if there are less or more than 10 ideas, need to trim or duplicate accordingly
            l = query_util.resize_list(l, self.config.num_samples)
        
            ideas[qr.problem_id] = l
        
        self.curr_step += 1

        return ideas

    def run(self):
        idea_queries = self.get_idea_queries()
        ideas = self.gen_and_extract_ideas(idea_queries)

        for phase in range(self.config.num_phases):
            if phase == 0:
                queries = self.get_init_queries(ideas)
            else:
                # use the saved solutions to build queries for the next phase (branching in the parallel tree search)
                queries = self.get_next_queries()

            if len(queries) == 0:
                print(f"All tasks have reached max budget, exiting")
                break

            # IMPORTANT: clear this right after the call to get_next_queries, since these phase solutions
            # may be used to seed the following set of queries
            for problem_id in self.phase_solutions.keys():
                self.phase_solutions[problem_id] = []
                self.phase_solutions_by_branch[problem_id] = {}

            # add 1 to max_fix_attempts since we need to include the initial attempt too
            for fix_iter in range(self.config.max_fix_attempts + 1):
                print(f"======================= phase: {phase}, fix iter: {fix_iter} =======================")
                new_samples = self.gen_samples(queries)
                eval_data = self.run_eval(new_samples)
                self.save_solutions(eval_data)
                queries = self.get_direct_fix_queries(eval_data)
                if len(queries) == 0:
                    print(f"======== All solutions passing/failed, exiting at phase: {phase}, fix iter: {fix_iter} ========")
                    break
            
            print(f"\n\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            print(f"Phase {phase} complete.")
            for problem_id, l in self.phase_solutions.items():
                print(f"  Task: {problem_id}, solution count: {len(l)}")

            print(f"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n")

            self.curr_phase += 1
            self.curr_step = 0


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    print(f"Starting Batch Generation with config: {config}")

    tree_search = ParallelTreeSearch(config)
    asyncio.run(tree_search.init())
    tree_search.run()


if __name__ == "__main__":
    main()

