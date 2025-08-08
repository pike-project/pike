# KernelBench Agent Framework

This documentation is broken down into the following sections:

- `agent_framework` - agent framework details
- `containers` - container options and design choices
- `eval_worker` - task evaluation details
- `profiling` - how to profile solutions with NVIDIA tools
- `hpc` - notes for particular HPC clusters
- `troubleshooting` - common issues
- `misc` - any other notes that could be useful

## Eval Worker and Agents

How to start the eval worker:

```bash
mkdir -p worker_io/input && mkdir -p worker_io/output && mkdir -p worker_io/scratch
python scripts/start_eval_worker.py --input_dir worker_io/input --output_dir worker_io/output --scratch_dir worker_io/scratch
```

How to run parallel tree search:

```bash
python scripts/parallel_tree_search.py data_dir=./data dataset_src=local server_type=cborg model_name=lbl/llama num_workers=50 worker_input_dir=worker_io/input worker_output_dir=worker_io/output level=1 task_start=1 task_end=2 num_samples=10

python scripts/parallel_tree_search.py data_dir=./data dataset_src=local server_type=cborg model_name=google/gemini-pro num_workers=50 worker_input_dir=worker_io/input worker_output_dir=worker_io/output level=1 task_start=1 task_end=2 num_samples=10

python scripts/parallel_tree_search.py data_dir=./data dataset_src=local server_type=google model_name=gemini-2.5-pro num_workers=50 worker_input_dir=worker_io/input worker_output_dir=worker_io/output level=1 task_start=1 task_end=1 num_samples=1
```

How to use the new eval script for a single task

```bash
python scripts/eval.py --level 3 --task 27 --code_path KernelBench/level3/27_RegNet.py --gpu_locks_dir worker_io/gpu_locks

python scripts/eval.py --level 1 --task 1 --code_path results/o3-test1/generated_kernel_level_1_problem_1.py --gpu_locks_dir worker_io/gpu_locks
```
