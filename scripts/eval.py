import os
import torch
import argparse
from torch.utils.cpp_extension import load
from torch.utils._pytree import tree_map
import importlib.util
from torch.utils.benchmark import Timer
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import inspect
from triton.testing import do_bench
import torch._logging
import logging
import warnings
import json

# it seems pytorch Timer timeit is much more noisy for small input sizes
# so we should always be using Triton
TIME_WITH_TRITON = True

# Both of these values are in ms
# note that if the benchmark ends up running longer than this for one repetition,
# at least one repetition will always complete
TRITON_BENCH_TIME_GOAL = 10000 # 10 seconds
TRITON_BENCH_WARMUP = 1000 # 1 second

# this is the number of reps that pytorch Timer timeit does
# Note that pytorch also does some warmup runs before starting these repetitions
TORCH_TIMER_REPS = 10000 # 10000 reps

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
    if compile:
        if module_fn is None:
            model = torch.compile(model, mode="max-autotune")
        else:
            module_fn = torch.compile(module_fn, mode="max-autotune")

    model_invocation = "model(*inputs)"
    if module_fn is not None:
        model_invocation = "model(*inputs, module_fn)"

    # TODO: this is where we should ensure we have exclusive access over the current GPU via some lock mechanism
    # METR wackiness, as sometimes their best solution is just a function, not a Model class
    moved_model = model
    if hasattr(model, "to"):
        moved_model = model.to(device)

    if TIME_WITH_TRITON:
        with torch.no_grad():
            moved_inputs = easy_to_device(inputs, device)

            if module_fn is None:
                bench = lambda: moved_model(*moved_inputs)
            else:
                bench = lambda: moved_model(*moved_inputs, module_fn)

            # the result here is in ms already
            runtime = do_bench(bench, rep=TRITON_BENCH_TIME_GOAL, warmup=TRITON_BENCH_WARMUP, return_mode='mean')
    else:
        timer = Timer(
            stmt=f"with torch.no_grad(): {model_invocation}",
            globals={
                "model": moved_model,
                "inputs": easy_to_device(inputs, device),
                "module_fn": module_fn
            },
        )
        runtime = timer.timeit(TORCH_TIMER_REPS).mean * 1000

    print(f"Name: {name}, compile: {compile}, runtime: {runtime:.3f} ms")

    return runtime


class Eval:
    def __init__(self, level, task, op_atol, op_rtol):
        self.level = level
        self.task = task

        self.op_atol = op_atol
        self.op_rtol = op_rtol

        self.task_id = f"{self.level}_{self.task}"

        task_str = f"{self.task}"
        if len(task_str) == 1:
            task_str = "00" + task_str
        elif len(task_str) == 2:
            task_str = "0" + task_str

        self.task_str = task_str

        # Initialize model and inputs
        self.device = torch.device("cuda:0")
        
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

    def init_models(self):
        # best_dir = Path.resolve(curr_path / "../data/best")

        # target_dir = best_dir / f"level_{self.level}" / self.task_str

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

        llm_path = Path.resolve(curr_path / f"../results/o3-test1/generated_kernel_level_1_problem_1.py")
        self.create_model("llm", llm_path)

    def get_model_output(self, name):
        model_data = self.models[name]

        model = model_data["model"]
        module_fn = model_data["module_fn"]
        
        # TODO: lock the GPU to ensure exclusive access, so we are not disrupting some other performance run
        if module_fn is None:
            return model.to(self.device)(*easy_to_device(self.inputs, self.device))
        else:
            return model.to(self.device)(*easy_to_device(self.inputs, self.device), module_fn)

    def check_correctness(self):
        # Test for correctness
        with torch.no_grad():
            baseline_output = self.get_model_output("baseline")

            for name in self.models.keys():
                if name == "baseline":
                    self.results[name]["correct"] = True
                    continue

                comp_output = self.get_model_output(name)

                correct = torch.allclose(
                    baseline_output.cpu(),
                    comp_output.cpu(),
                    rtol=self.op_rtol,
                    atol=self.op_atol,
                )

                max_diff = torch.max(torch.abs(baseline_output.cpu() - comp_output.cpu())).item()
                print(f"Tested {name} - Correct: {correct}, Max Diff: {max_diff}")

                self.results[name]["correct"] = correct
                self.results[name]["max_diff"] = max_diff

    def profile(self):
        with torch.no_grad():
            for name in self.models.keys():
                if name == "metr":
                    comp_output = self.get_model_output(name)
                    print(comp_output)

    def time_model(self, name, compile):
        model_data = self.models[name]
        model = model_data["model"]
        module_fn = model_data["module_fn"]
        model_idx = model_data["idx"]

        runtime = time_model(model, module_fn, self.inputs, self.device, compile, name)

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

        return runtime

    def collect_model_results(self, name, with_compile=True):
        # no reason to collect runtime results if correctness does not pass
        res = self.results[name]
        if not res["loaded"] or not res["correct"]:
            return

        runtimes = {}
        runtime_eager = self.time_model(name, False)
        runtimes["eager"] = runtime_eager

        if with_compile:
            runtime_compile = self.time_model(name, True)
            runtimes["compile"] = runtime_compile

        res["runtimes"] = runtimes

    def run(self):
        for name in self.models.keys():
            self.collect_model_results(name)

    def save_results(self):
        results_dir = Path.resolve(curr_path / "../results/tmp")

        os.makedirs(results_dir, exist_ok=True)

        results_path = results_dir / f"task_{self.task_id}.json"

        with open(results_path, "w") as f:
            json.dump(self.results, f)

    def plot(self):
        df = pd.DataFrame(self.runtime_results)

        f, ax = plt.subplots(1, figsize=(4, 3))

        labels = df["label"]
        values = df["runtime"]

        col = [
            "#2aadb6",
            "#ff6583",
            "#aa6fc5",
            "#ffa600",
        ]

        colors = []

        for _, row in df.iterrows():
            model_idx = row['model_idx']
            colors.append(col[model_idx])

        plt.title(f"Task: {self.task_id}")
        plt.ylabel('Runtime (ms)')

        plt.bar(labels, values, color=colors, linewidth=1, edgecolor='black')

        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y')
        plt.gca().set_axisbelow(True)

        plt.subplots_adjust(left=0.2, bottom=0.3)

        figs_dir = Path.resolve(curr_path / "../figs/eval_new_3")

        os.makedirs(figs_dir, exist_ok=True)

        plt.savefig(figs_dir / f"task_{self.task_id}.pdf")

    def create_model(self, name, file_path, input_changes=None, cuda_path=None, baseline=False):
        try:
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
        except Exception as e:
            print(f"Loading model {name} failed for task {self.task_id}")

            self.results[name] = {
                "loaded": False,
            }

        self.curr_model_idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_atol", type=float, default=1e-3)
    parser.add_argument("--op_rtol", type=float, default=1e-1)
    args = parser.parse_args()

    level = 1
    task = 1

    ev = Eval(level, task, args.op_atol, args.op_rtol)
    ev.init_models()
    ev.check_correctness()
    ev.collect_model_results("baseline", with_compile=True)
    ev.collect_model_results("llm", with_compile=False)
    # ev.save_results()

    # level = 3

    # # NOTE: task 3-36, 3-37, 3-39, 3-40, 3-42 in METR has modified inputs, so skip it for now
    # # METR 3-43 complains with invalid argument
    # # Sakana - something is broken with 3-47
    # skip_level_3 = [36, 37, 39, 40, 42, 43, 47]

    # incorrect_metr = [8, 13, 19, 22, 23, 41, 50]

    # for task in range(27, 28):
    #     print(f"Task: {level}-{task}")

    #     if level == 3:
    #         if task in skip_level_3:
    #             print(f"Skipping task: {level}-{task}")
    #             continue

    #     ev = Eval(level, task, args.op_atol, args.op_rtol)
    #     try:
    #         ev.init_models()
    #         ev.profile()
    #         # ev.check_correctness()
    #         # ev.run()
    #         # ev.save_results()
    #     except Exception as e:
    #         print(e)
    # # ev.plot()

if __name__ == "__main__":
    main()
