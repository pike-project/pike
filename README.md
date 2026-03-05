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

The simplest PIKE setup involves two components:

- A containerized evaluator which runs kernels on the target GPU
- A search script on the host machine which makes LLM queries and communicates with the evaluator via filesystem

We recommend setting up your host environment using `uv` ([uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))

Clone this repository, then do the following:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
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

**To run the Eval Worker:** install Docker and the NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

## Running PIKE

As noted above, running a PIKE search requires two components running simultaneously:

- **Eval Worker** — runs the evaluator in a container
- **Search process** — the PIKE-B or PIKE-O LLM optimization strategy

Start the Eval Worker first, then run the search in a second terminal.

### Start Eval Worker

Ensure Docker and NVIDIA Container Toolkit are installed, then start the containerized Eval Worker (this script will fetch the container from online if not yet installed):

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper> --max-active-tasks 20
```

### Run the Search

`scripts/run_search.py` is a unified entry point for PIKE-B and PIKE-O. It automatically manages the eval HTTP server internally, so no separate server setup is needed. The `--port` flag selects which local port the internal server listens on — any available port works.

Try a dry run first (does not require the Eval Worker to be running):

```bash
python scripts/run_search.py \
    --output-dir data/pike-data \
    --strategy pike-b \
    --level 3-pike \
    --server-type google \
    --model-name gemini-2.5-pro \
    --run-name h100_level_3-pike_pike-b \
    --task-start 1 --task-end 50 \
    --dry-run
```

The dry run simulates eval responses without hitting the worker. Once satisfied, run without `--dry-run` (Eval Worker must be running).

For PIKE-O, pass `--strategy pike-o`. The script will clone and install [pike-openevolve](https://github.com/pike-project/pike-openevolve) automatically. This strategy does not currently have a dry run mode.

For advanced setups (running components separately, remote eval server), see [`docs/advanced_setup.md`](docs/advanced_setup.md).

## Documentation

Additional documentation is available in the [`docs/`](docs/) directory, covering the eval worker, containers, HPC cluster setup, LLM API setup, profiling, and troubleshooting.

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
