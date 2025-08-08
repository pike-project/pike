# KernelBench Agent Framework

Our KernelBench agent framework builds off of KernelBench. We chose to fork the original KernelBench repository, consolidating the agent framework implementation with the benchmark set that our framework is designed for. The agent framework could, in principle, be decoupled to operate on a more general set of tasks, but this is not currently our primary goal. Thus, we chose simplicity over modularity, ensuring that the agent framework is well-tuned to the benchmark we are optimizing for.

To read about KernelBench itself, rather than our agent framework docs, see: https://github.com/ScalingIntelligence/KernelBench

## Setup

Clone this repository, then do the following:

```
conda create --name kernel-bench python=3.12
conda activate kernel-bench
pip install -r requirements.txt
pip install -e .

# additional data analysis
pip install matplotlib pandas scipy
```

Set API keys to environment variables, e.g. `export OPENAI_API_KEY=<...>`, ` export GEMINI_API_KEY=<...>`, etc.

## Start Agent Framework

To start running the agent framework, first try a dry run (does not require the eval worker):

```bash
python -u scripts/parallel_tree_search.py data_dir=./data server_type=google model_name=gemini-2.5-pro num_workers=30 worker_input_dir=./worker_io/input worker_output_dir=./worker_io/output level=0 task_start=1 task_end=5 num_samples=10 num_phases=5 max_fix_attempts=5 dry_run=True
```

(**note:** a full reference script to run the agent framework, then do analysis aftewards, can be found in `./tools/run.sh`)

If this works fine, you can switch to `dry_run=False`. Now, the agent framework will wait until the eval worker is running before it does anything.

## Start Eval Worker

If you are working on a machine where you have root access, install Docker, along with the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Once the agent framework is waiting for an eval worker, start the containerized eval worker in a different window, passing in the correct GPU architecture:

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper>
```

If you are on an HPC system, a container engine like Docker, Podman, or Apptainer is likely available. To start the eval worker in a separate job, you can reference existing worker job scripts in `tools/worker_jobs/`

Currently supported HPC systems:
- NERSC Perlmutter via podman-hpc, Ampere A100: `tools/worker_jobs/perlmutter.sh`
- LBL Lawrencium via apptainer, Hopper H100: `tools/worker_jobs/lrc.sh`

As soon as the eval worker job starts, the waiting agent framework script will connect to it via NFS and start sending it eval tasks.

## Eval-Only (No Agent Framework)

If you only want to time a particular set of solutions, without running the agent framework, you can do so like this:

```bash
python scripts/solution_eval/eval_solutions.py --level 3 --solutions baseline --mode <eager/compile>
```

After you run this, start the eval worker in a separate window and the eval tasks will be sent there.

## Documentation

To learn more about using our agent framework, see `docs/README.md`
