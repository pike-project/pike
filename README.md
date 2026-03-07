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

- A containerized evaluator which runs PyTorch/kernel code on the target GPU
- A script running on the host which sends code for evaluation to the evaluator via filesystem. This is either:
    - An LLM-driven search script
    - A baseline script to evaluate pre-existing baseline PyTorch/kernel code

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
export OPENROUTER_API_KEY=<...>
```

Then source the changes via:

```bash
source ~/.bashrc
```

**To run the Eval Worker:** install Docker and the NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

## Running PIKE

As noted above, running a PIKE search requires two components running simultaneously:

- **Eval Worker** — runs the evaluator in a container
- **Search/Baseline process** — The optimization search process, or baseline manager script

Start the Eval Worker first, then run the search in a second terminal.

### Dry Run

Try a dry run first to test the host components (does not require the Eval Worker to be running):

```bash
./tools/dry_run.sh
```

If everything worked correctly, you should figures with bogus data in `data/dry-run/pike-out/h100_level_3-pike/results/figs`.

### Start Eval Worker

Ensure Docker and NVIDIA Container Toolkit are installed, then start the containerized Eval Worker (this script will fetch the container from online if not yet installed):

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper> --max-active-tasks 10 --pull-image
```

If you need to restart the worker in the middle of a run, clear any outstanding messages via `rm -rf worker_io`

### Evaluate Baselines

This step collects runtimes for the original PyTorch code, allowing calculation of speedups. This step can be run before or after the search, but both steps MUST happen before generating figures.

```bash
python scripts/eval_baselines.py --output-dir data/pike-data --level 3-pike
```

This script and the `run_search` script below run an eval HTTP server that is for internal use only. The default port for this internal server is 8000, but this can be adjusted with the `--port` flag (any available port should work fine).

### Run the Search

Keep the Eval Worker running for this. The search process submits PyTorch/kernel code to the evaluator, just like in the Evaluate Baselines step.

**Important:** use the same `--output-dir` here as you used for baseline evaluation, so that `generate_figs.py` can find both the search results and the baseline runtimes in one place.

```bash
python scripts/run_search.py --run-name <run_name> --output-dir data/pike-data --strategy pike-b --level 3-pike --server-type google --model-name gemini-2.5-pro --task-start 1 --task-end 50
```

Set desired server type (e.g. google, openai, openrouter), and model name (e.g. `gemini-2.5-pro`, `gpt-oss-120b`)

You can select any run name for your run, passed in via `--run-name`. The output for the run will then appear in `<output-dir>/full-pike-runs/level_<level>/<run_name>`. If a run fails or you kill a run early, it is highly recommended to rename/remove that failed run, or change the `--run-name` value before restarting the run.

For PIKE-O, pass `--strategy pike-o`. The script will clone and install [pike-openevolve](https://github.com/pike-project/pike-openevolve) automatically.

### Generate Figures

After the search and the baseline evaluation complete, generate figures for the run:

```bash
python scripts/generate_figs.py --input-dir data/pike-data --output-dir data/pike-out
```

### Original Paper Figure Generation

The original data from the paper is available here: https://huggingface.co/datasets/knagaitsev/pike-data-compressed

The main original figures from the paper can easily be generated by fetching this data, then running the figure generation script on the data:

```bash
# fetching will take some time and requires ~80 GB on disk
python scripts/fetch_paper_data.py

# the fetch script places the data in data/paper-data
python scripts/generate_figs.py --input-dir data/paper-data/pike-data --output-dir data/paper-data/pike-out --paper
```

The `--paper` option should only be used on original paper data, as it only includes a subset of results in some plots, and adds additional markings to plots.

## Documentation

Additional documentation is available in the [`docs/`](docs/) directory, covering the eval worker, containers, HPC cluster setup, LLM API setup, profiling, and troubleshooting.

For advanced setups (running components separately, remote eval server), see [`docs/advanced_setup.md`](docs/advanced_setup.md).

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
