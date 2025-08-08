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

To set up the Docker container and start the eval worker in that container:

```bash
./sandbox/tools/build-deps.sh

# to push the built image (if you need to)
podman-hpc login docker.io
podman-hpc tag kernel-bench-deps:latest docker.io/<username>/kernel-bench-deps:v0.1
podman-hpc push docker.io/<username>/kernel-bench-deps:v0.1

python sandbox/tools/start_worker_container.py
```

Apptainer:

```bash
apptainer registry login --username loonride docker://docker.io
APPTAINER_TMPDIR=$SCRATCH/apptainer/tmp APPTAINER_CACHEDIR=$SCRATCH/apptainer/cache apptainer pull kernel-bench-deps.sif docker://docker.io/<username>/kernel-bench-deps:v0.3
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

Migrate a Docker container:

```
podman-hpc migrate <name>
```

Attach to running docker container:

```bash
podman-hpc exec -it <id> bash
```
