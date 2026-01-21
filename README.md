<p align="center">
  <img src="https://raw.githubusercontent.com/pike-project/pike/main/assets/logo-rounded.png" width="160">
</p>

# PIKE

A framework for comparing multi-agent PyTorch optimization systems, along with multiple optimization strategy implementations.

These components are collectively defined as PyTorch Inference Kernel Evolution (PIKE).

See the paper preprint here: https://arxiv.org/abs/2511.16964

## About

This is a fork of [KernelBench](https://github.com/ScalingIntelligence/KernelBench) by Anne Ouyang, Simon Guo, and Azalia Mirhoseini. Benchmark additions and modifications are included from [KernelBenchFiltered](https://github.com/METR/KernelBenchFiltered) by METR.

This repository contains:

- a refined set of KernelBench benchmarks
- our evaluator setup
- PIKE-B, a multi-agent evolutionary branching strategy for PyTorch optimization

The implementation for PIKE-O can be found in the [pike-openevolve](https://github.com/pike-project/pike-openevolve) repository. PIKE-O is an OpenEvolve-based PyTorch optimization strategy. It makes use of the evaluator in this repository.

<!-- # KernelBench Agent Framework

We chose to fork the original KernelBench repository, consolidating the agent framework implementation with the benchmark set that our framework is designed for. The agent framework could, in principle, be decoupled to operate on a more general set of tasks, but this is not currently our primary goal. Thus, we chose simplicity over modularity, ensuring that the agent framework is well-tuned to the benchmark we are optimizing for. -->

## Setup

We recommend setting up your environment using `uv` ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

Clone this repository, then do the following:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

# additional data analysis
uv pip install matplotlib pandas scipy
```

Save the following API key environment variables to `~/.bashrc`:

```bash
export OPENAI_API_KEY=<...>
export GEMINI_API_KEY=<...>
```

Then source the changes via:

```bash
source ~/.bashrc
```

## Running PIKE

Running a PIKE implementation involves 3 key components. It is recommended to start the components in the order listed below.

- Eval Worker: Runs evaluator in a container, and allows low-level, filesystem-based communication with the containerized worker
- Eval Server: Exposes HTTP server for sending and receiving eval data, managing the low-level communication with the Eval Worker internally
- PIKE Implementation (PIKE-B/PIKE-O): implements the LLM-based optimization strategy

## Start Eval Worker

If you are working on a machine where you have root access, install Docker, along with the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Start the containerized Eval Worker like so, passing in the correct GPU architecture:

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper> --max-active-tasks 20
```

<!-- python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper> -->

<!-- If you are on an HPC system, a container engine like Docker, Podman, or Apptainer is likely available. To start the eval worker in a separate job, you can reference existing worker job scripts in `tools/worker_jobs/`

Currently supported HPC systems:
- NERSC Perlmutter via podman-hpc, Ampere A100: `tools/worker_jobs/perlmutter.sh`
- LBL Lawrencium via apptainer, Hopper H100: `tools/worker_jobs/lrc.sh`

As soon as the eval worker job starts, the waiting agent framework script will connect to it via NFS and start sending it eval tasks. -->

## Start Eval Server

Once the Eval Worker is started, start the Eval Server. The Eval Server is an HTTP server that acts as a proxy between the Eval Worker's low-level communication channel and the PIKE implementation eval requests.

```bash
python scripts/disk_channel_server.py --port 8000
```

## Run PIKE-B

To run PIKE-B directly, first try a dry run (does not require the eval worker):

```bash
python scripts/parallel_tree_search.py --server-type google --model-name gemini-2.5-pro --level 3-pike --task-start 1 --task-end 50 --num-branches 10 --max-fix-attempts 5 --query-budget 300 --eval-port 8000 --dry-run --run-dir <path/to/output-dir>
```

The dry run simulates a series of phases in which some evals fail, and others succeed with random runtime values. If this works fine, you can remove `--dry-run`. Do a real run only after the Eval Worker and Eval Server are running.

## Run PIKE-O

First, clone the following repository: [pike-openevolve](https://github.com/pike-project/pike-openevolve)

In the pike-openevolve directory:

```bash
pip install -e .
```

As with PIKE-B, run the following (from within the pike-openevolve directory) only after the Eval Worker and Eval Server are running:

```bash
python examples/kernelbench/run.py --pike-dir <path/to/this-repo> --level 3-pike --task-start 1 --task-end 50 --max-fix-attempts 5 --eval-port 8000 --run-dir <path/to/output-dir>
```

To further tune the PIKE-O system configuration, edit `examples/kernelbench/config.yaml`

## Documentation

To learn more about using PIKE, see `docs/README.md`

## Citation

```bibtex
@misc{nagaitsev2025pike,
    title={Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems}, 
    author={Kirill Nagaitsev and Luka Grbcic and Samuel Williams and Costin Iancu},
    year={2025},
    eprint={2511.16964},
    archivePrefix={arXiv},
    primaryClass={cs.MA},
    url={https://arxiv.org/abs/2511.16964}, 
}
```
