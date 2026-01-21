## Eval-Only (No Agent Framework)

### Single-Solution Eval

How to use the eval script for a single task (without container or HPC job allocation):

```bash
python scripts/eval.py --level 3 --task 27 --code-path KernelBench/level3/27_RegNet.py --gpu-locks-dir worker_io/gpu_locks --compile
```

Include the `--compile` flag if you want to the eval to run with `torch.compile` enabled

### Multi-Solution Eval

If you only want to time a particular set of solutions, without running the agent framework, you can do so like this:

```bash
# baseline solutions
python scripts/solution_eval/eval_solutions.py --level 0 --solutions baseline --mode <eager/compile>

# agent-generated solutions, must pass in the path to the run dir
python scripts/solution_eval/eval_solutions.py --level 0 --solutions agent --run-dir <run_dir> --mode eager
```

After you run this, start the eval worker in a separate window and the eval tasks will be sent there.
