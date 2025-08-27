import os
import gc
import torch
import argparse
from torch.utils.cpp_extension import load
from torch.utils._pytree import tree_map
import importlib.util
from torch.utils.benchmark import Timer
from pathlib import Path
import inspect
from triton.testing import do_bench
import torch._logging
import logging
import warnings
import json
import time
import uuid
import filelock
from filelock import FileLock

# Both of these values are in ms
# note that if the benchmark ends up running longer than this for one repetition,
# at least one repetition will always complete
TRITON_BENCH_TIME_GOAL = 5000 # 5 seconds
TRITON_BENCH_WARMUP = 1000 # 1 second

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))


def easy_to_device(pytree, device):
    return tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, pytree
    )


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_model(model, module_fn, inputs, device, compile, name):
    model_invocation = "model(*inputs)"
    if module_fn is not None:
        model_invocation = "model(*inputs, module_fn)"

    # METR wackiness, as sometimes their best solution is just a function, not a Model class
    moved_model = model
    if hasattr(model, "to"):
        moved_model = model.to(device)
    
    moved_inputs = easy_to_device(inputs, device)

    if compile:
        if module_fn is None:
            moved_model = torch.compile(moved_model, mode="max-autotune")
        else:
            module_fn = torch.compile(module_fn, mode="max-autotune")

    with torch.no_grad():
        if module_fn is None:
            bench = lambda: moved_model(*moved_inputs)
        else:
            bench = lambda: moved_model(*moved_inputs, module_fn)

        # the result here is in ms already
        runtime = do_bench(bench, rep=TRITON_BENCH_TIME_GOAL, warmup=TRITON_BENCH_WARMUP, return_mode='mean')

    print(f"Name: {name}, compile: {compile}, runtime: {runtime:.3f} ms")

    return runtime


class Eval:
    def __init__(self, level, task, op_atol, op_rtol, gpu_locks_dir=None):
        self.level = level
        self.task = task
        self.op_atol = op_atol
        self.op_rtol = op_rtol
        self.gpu_locks_dir = gpu_locks_dir

        self.task_id = f"{self.level}_{self.task}"

        self.num_gpus = torch.cuda.device_count()

        print(f"GPU count: {self.num_gpus}")

        self.locked_device = None
        self.held_lock = None

        self.devices = []
        self.gpu_locks = []

        for gpu_id in range(self.num_gpus):
            self.devices.append(torch.device(f"cuda:{gpu_id}"))
            if self.gpu_locks_dir is not None:
                gpu_lock_path = self.gpu_locks_dir / f"gpu_{gpu_id}.lock"
                self.gpu_locks.append(FileLock(gpu_lock_path))
        
        # this is very dicey, better way would be to pass in
        # the weights for the Linear layers
        self.input_seed = 0
        self.weights_seed = 1

        # batch_size = 1000
        # self.input_changes = {
        #     "batch_size": batch_size,
        # }
        self.input_changes = {}

        self.inputs = None
        self.init_inputs = None

        self.curr_model_idx = 0
        self.models = {}

        self.results = {}
        self.runtime_results = []

    def create_baseline_model(self):
        baseline_dir = Path.resolve(curr_path / f"../KernelBench/level{self.level}")

        baseline_path = None

        for filename in os.listdir(baseline_dir):
            if not filename.endswith(".py"):
                continue

            file_task = int(filename.split("_")[0])
            if file_task == self.task:
                baseline_path = baseline_dir / filename
                break

        # functional_path = target_dir / "pytorch_functional.py"
        # kernel_path = target_dir / "kernel.cu"
        # metr_path = Path.resolve(curr_path / f"../../KernelBenchFiltered/best_agent_solutions/level_{self.level}/task_{self.task}.py")

        # important: must do baseline first
        self.create_model("baseline", baseline_path, baseline=True)
        # self.create_model("sakana-functional", functional_path)
        # self.create_model("sakana-cuda", functional_path, cuda_path=kernel_path)
        # self.create_model("metr", metr_path)

    def get_model_device(self, name):
        model_data = self.models[name]

        model = model_data["model"]

        assert self.held_lock is not None, "Lock should be held"
        assert self.locked_device is not None, "Locked device should exist"
        
        # make sure we are not disrupting some other performance run by ensuring we have a lock while in this region
        return model.to(self.locked_device)

    def get_model_output(self, name, compile=False):
        model_data = self.models[name]
        module_fn = model_data["module_fn"]

        model_device = self.get_model_device(name)

        if compile:
            model_device = torch.compile(model_device, mode="max-autotune")
        
        # make sure we are not disrupting some other performance run by ensuring we have a lock while in this region
        if module_fn is None:
            return model_device(*easy_to_device(self.inputs, self.locked_device))
        else:
            return model_device(*easy_to_device(self.inputs, self.locked_device), module_fn)

    # returns all_correct, max_diffs list
    def compare_output(self, baseline_output, comp_output):
        all_correct = True
        max_diffs = []

        if not isinstance(baseline_output, tuple):
            baseline_output = (baseline_output,)
            comp_output = (comp_output,)

        for idx, (b, c) in enumerate(zip(baseline_output, comp_output)):
            correct = torch.allclose(
                b.cpu(),
                c.cpu(),
                rtol=self.op_rtol,
                atol=self.op_atol,
            )

            max_diff = torch.max(torch.abs(b.cpu() - c.cpu())).item()

            max_diffs.append(max_diff)

            if not correct:
                all_correct = False
        
        return all_correct, max_diffs

    def check_correctness(self):
        start_time = time.time()

        # Test for correctness
        with torch.no_grad():
            for name in self.models.keys():
                if name == "baseline":
                    self.results[name]["correct"] = True
                    continue

                comp_output = self.get_model_output(name)

                # get this model output after the LLM-generated model output to avoid any cheating from
                # lingering values in memory
                baseline_output = self.get_model_output("baseline")

                correct, max_diff = self.compare_output(baseline_output, comp_output)
                print(f"Tested {name} - Correct: {correct}, Max Diff: {max_diff}")

                self.results[name]["correct"] = correct
                self.results[name]["max_diff"] = max_diff

        end_time = time.time()
        print(f"Time to check correctness: {end_time - start_time:.2f}s")

    def profile(self, compile):
        with torch.no_grad():
            for name in self.models.keys():
                if name == "llm":
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/torch"),
                        record_shapes=True,
                        with_stack=True,
                        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
                    ) as prof:
                        comp_output = self.get_model_output(name, compile=compile)
                    print(comp_output)

                    prof.export_stacks("./log/torch/stacks_gpu.out", metric='self_cuda_time_total')

                    # prof_table = str(prof.key_averages().table(
                    #     sort_by="cpu_time_total",
                    #     row_limit=20,
                    # ))
                    # with open("./log/torch/table.log", "w") as f:
                    #     f.write(prof_table)

    def time_model(self, name, compile):
        start_time = time.time()

        model_data = self.models[name]
        model = model_data["model"]
        module_fn = model_data["module_fn"]
        model_idx = model_data["idx"]

        # this is where we should ensure we have exclusive access over the current GPU via lock mechanism
        assert self.held_lock is not None, "Lock should be held"
        assert self.locked_device is not None, "Locked device should exist"

        runtime = time_model(model, module_fn, self.inputs, self.locked_device, compile, name)

        label = name + "-"
        if compile:
            label += "compile"
        else:
            label += "eager"

        self.runtime_results.append({
            "task_id": self.task_id,
            "label": label,
            "runtime": runtime,
            "model_idx": model_idx,
        })

        end_time = time.time()
        print(f"Time to eval model {name}, compile={compile}: {end_time - start_time:.2f}s")

        return runtime

    def collect_model_results(self, name, compile):
        # no reason to collect runtime results if correctness does not pass
        res = self.results[name]
        if not res["loaded"] or not res["correct"]:
            return

        res["runtime"] = self.time_model(name, compile)

    def run(self):
        for name in self.models.keys():
            self.collect_model_results(name)

    def save_results(self, results_path: Path):
        # results_dir = Path.resolve(curr_path / "../results/tmp")

        # os.makedirs(results_dir, exist_ok=True)

        # results_path = results_dir / f"task_{self.task_id}.json"

        with open(results_path, "w") as f:
            json.dump(self.results, f)

    def create_model(self, name, file_path, input_changes=None, cuda_path=None, baseline=False):
        start_time = time.time()

        task = load_module_from_path(file_path)

        if baseline:
            torch.manual_seed(self.input_seed)

            input_changes = self.input_changes
            if input_changes is not None:
                for key, val in input_changes.items():
                    setattr(task, key, val)
            
            # METR wackiness
            inputs = []
            if hasattr(task, "get_inputs"):
                inputs = task.get_inputs()

            init_inputs = None
            # METR wackiness
            if hasattr(task, "get_init_inputs"):
                init_inputs = task.get_init_inputs()

            self.inputs = inputs
            self.init_inputs = init_inputs   

        torch.manual_seed(self.weights_seed)

        if hasattr(task, "ModelNew"):
            model = task.ModelNew(*self.init_inputs)
        else:
            model = task.Model(*self.init_inputs)

        model.eval()

        has_module_fn = hasattr(task, "module_fn")
        module_fn = None
        if has_module_fn:
            module_fn = task.module_fn
        
        if cuda_path is not None:
            module_fn = load(
                name="forward",
                sources=[cuda_path],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=True,
            ).forward

        self.models[name] = {
            "model": model,
            "module_fn": module_fn,
            "idx": self.curr_model_idx
        }

        self.results[name] = {
            "loaded": True,
        }

        self.curr_model_idx += 1

        end_time = time.time()
        print(f"Load time for model {name}: {end_time - start_time:.2f}s")

    def acquire_available_gpu_lock(self, check_interval: float = 1.0) -> tuple[int, FileLock]:
        """
        Waits until one of the GPU locks is available, acquires it, and returns (gpu_id, lock).
        """
        if self.gpu_locks_dir is None:
            lock_path = f"/tmp/{uuid.uuid4()}"
            lock = FileLock(lock_path)
            lock.acquire(timeout=0)

            return self.devices[0], lock

        while True:
            for i, lock in enumerate(self.gpu_locks):
                try:
                    lock.acquire(timeout=0)  # Non-blocking try-lock
                    return self.devices[i], lock
                except filelock.Timeout:
                    continue
            # No GPU lock acquired; wait and retry
            time.sleep(check_interval)

    def acquire_gpu_lock(self):
        device, lock = self.acquire_available_gpu_lock()
        self.locked_device = device
        self.held_lock = lock

        torch.cuda.set_device(device)

    def release_gpu_lock(self):
        self.held_lock.release()

    def print_gpu_mem(self):
        mem_alloc = torch.cuda.memory_allocated() / 1e6
        mem_res = torch.cuda.memory_reserved() / 1e6

        print(f"CUDA memory allocated at start: {mem_alloc} MB")
        print(f"CUDA memory reserved at start: {mem_res} MB")

    def free_models_and_inputs(self):
        del self.inputs
        del self.models

    def cleanup(self):
        self.free_models_and_inputs()
        torch.cuda.empty_cache()

        # Collect unused Python objects
        gc.collect()

        torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser()
    # NOTE: must be str because we could pass in variations of a level number like "3-metr"
    parser.add_argument("--level", type=str)
    parser.add_argument("--task", type=int)
    parser.add_argument("--code_path", type=str)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--gpu_locks_dir", type=str, required=False)
    parser.add_argument("--op_atol", type=float, default=1e-2)
    parser.add_argument("--op_rtol", type=float, default=1e-2)
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--profile", action='store_true')
    args = parser.parse_args()

    level = args.level
    task = args.task

    # llm_path = Path.resolve(curr_path / f"../results/o3-test1/generated_kernel_level_1_problem_1.py")
    llm_path = args.code_path

    output_path = None
    if args.output_path is not None:
        output_path = Path(args.output_path)
    
    gpu_locks_dir = None
    if args.gpu_locks_dir is not None:
        gpu_locks_dir = Path(args.gpu_locks_dir)
        os.makedirs(gpu_locks_dir, exist_ok=True)

    ev = Eval(level, task, args.op_atol, args.op_rtol, gpu_locks_dir=gpu_locks_dir)
    
    ev.create_baseline_model()
    ev.create_model("llm", llm_path)

    ev.acquire_gpu_lock()

    if args.profile:
        ev.profile(args.compile)
    else:
        # sanity check print, make sure the GPU is completely free to work with
        ev.print_gpu_mem()

        # HACK: do not actually release gpu lock, just let it get released by process exiting
        ev.check_correctness()
        ev.collect_model_results("llm", compile=args.compile)
        ev.cleanup()

        # try:
        #     ev.check_correctness()
        #     ev.collect_model_results("llm", compile=True)
        #     ev.cleanup()
        # finally:
        #     ev.release_gpu_lock()

        if output_path is not None:
            ev.save_results(output_path)

if __name__ == "__main__":
    main()
