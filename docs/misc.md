# Miscellaneous Notes

## Package Setup

We probably want a `setup.py` like this eventually:

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

## Old Tool Notes (Likely Obsolete)

```bash
python -u scripts/start_eval_worker.py --input_dir ./data/worker_io/input --output_dir ./data/worker_io/output --scratch_dir ./data/worker_io/scratch --arch <Ampere/Hopper>
```

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
