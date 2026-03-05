## Eval-Only (No Agent Framework)

### Single-Solution Eval

How to use the eval script for a single task (without container or HPC job allocation):

```bash
python scripts/eval.py --level 3 --task 27 --code-path KernelBench/level3/27_RegNet.py --gpu-locks-dir worker_io/gpu_locks --compile
```

Include the `--compile` flag if you want to the eval to run with `torch.compile` enabled

### Baseline Evals

To collect all baseline runtimes (eager, compile, tensorrt, metr) in the pike-data format consumed by `scripts/generate_figs.py`, use `scripts/eval_baselines.py` (starts the eval HTTP server automatically):

```bash
python scripts/eval_baselines.py --output-dir data/pike-data --level 3-pike
```

This writes `eager.json`, `compile.json`, `tensorrt.json`, and `metr.json` to `data/pike-data/baseline-runtimes/h100_level_3-pike/`.

### Lower-Level Multi-Solution Eval

For finer control, `scripts/solution_eval/eval_solutions.py` can be used directly. The Eval Worker and Eval HTTP Server must both be running first (see [`advanced_setup.md`](advanced_setup.md)).

```bash
# baseline solutions
python scripts/solution_eval/eval_solutions.py --level 3-pike --solutions baseline --mode <eager/compile/tensorrt>

# agent-generated solutions, must pass in the path to the run dir
python scripts/solution_eval/eval_solutions.py --level 3-pike --solutions agent --run-dir <run_dir> --mode eager
```
