## Requirements

Set up local environment for profiling:

```bash
conda install -c nvidia nsight-compute
pip install -r requirements.txt
pip install -e .

# to profile
# (disable any profiling tools such as DCGMI currently using the GPU)
dcgmi profile --pause
ncu --set full --export output2.ncu-rep python example.py
```

Note: For some reason the pytorch inline CUDA compiler uses `c++` executable, not `g++`. Make sure to do the following to allow pytorch to compile the inline CUDA code:

```bash
mkdir -p $HOME/bin
ln -sf "$(which g++)" $HOME/bin/c++

# add to .bashrc
export PATH=$HOME/bin:$PATH
```

For data analysis:

```bash
pip install matplotlib pandas
```

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

**Important:** to run on a particular architecture, you need to make sure that places which reference this architecture are set correctly (e.g. A100/Ampere)







## Old Tool Notes

How to generate samples:

```bash
python3 scripts/generate_samples.py data_dir=./data dataset_src=local level=1 server_type=cborg model_name=lbl/llama num_workers=50
```

Single run as a test:

```bash
python3 scripts/generate_and_eval_single_sample.py dataset_src=local level=1 problem_id=1 server_type=cborg model_name=lbl/llama log_generated_kernel=True log_prompt=True log=True

# to just run and check the result
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=./KernelBench/level1/1_Square_matrix_multiplication_.py level=1 problem_id=1 kernel_src_path=./results/eval_logs/generated_kernel_level_1_problem_1.py
# python3 scripts/run_and_check.py ref_origin=kernelbench level=1 problem_id=1 kernel_src_path=./results/eval_logs/generated_kernel_level_1_problem_1.py
```

This can be run outside of the Docker container:

```bash
python3 scripts/generate_samples.py data_dir=./data run_name=test1 dataset_src=local level=1 server_type=cborg model_name=lbl/llama num_workers=50
```

This should be run within the Docker container:

```bash
python3 scripts/eval_from_generations.py data_dir=/data run_name=test1 dataset_src=local level=1 num_gpu_devices=4 timeout=300
```

Migrate a Docker container:

```
podman-hpc migrate <name>
```

Attach to running docker container:

```bash
podman-hpc exec -it <id> bash
```

TODO: we probably want a `setup.py` like this:

```python
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="kernel-bench",
        version="0.0.1",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
```

If you need to clear all cache info (to make sure there is no docker weirdness when mounting this directory to the container):

```bash
rm -rf build dist **/*.egg-info **/__pycache__
```

## Prompting Notes

We should pass this back to the correctness fixing LLM, so that it knows it is allowed to print:

```
You are allowed to print information in the code at intermediate steps, as the stdout can be used to resolve correctness issues if they exist.
```

However, if correctness is achieved, a final pass should be made to remove this debugging stuff.
