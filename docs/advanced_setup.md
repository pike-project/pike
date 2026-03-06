# Advanced Setup

This document describes how to run the PIKE pipeline components separately. The standard `scripts/run_search.py` workflow (described in the main README) handles all of this automatically, so you only need this if you want finer control over the individual components.

## Components

The PIKE pipeline has three components:

1. **Eval Worker** — containerized GPU evaluator, communicates over the filesystem
2. **Eval HTTP Server** — HTTP proxy between the worker's filesystem channel and the search process (`scripts/disk_channel_server.py`)
3. **Search Process** — PIKE-B (`scripts/parallel_tree_search.py`) or PIKE-O (`pike-openevolve`)

Start them in the order listed below.

## Start Eval Worker

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper> --max-active-tasks 20
```

## Start Eval HTTP Server

Once the Eval Worker is running, start the Eval HTTP Server. It acts as a proxy between the worker's low-level filesystem channel and the search process:

```bash
python scripts/disk_channel_server.py --port 8000
```

## Run PIKE-B

```bash
python scripts/parallel_tree_search.py --server-type google --model-name gemini-2.5-pro --level 3-pike --task-start 1 --task-end 50 --num-branches 10 --max-fix-attempts 5 --query-budget 300 --eval-port 8000 --run-dir <path/to/output-dir>
```

Add `--dry-run` to simulate eval responses without hitting the worker.

After a PIKE-B run completes, the output is in the original PIKE-B format. To convert it to the openevolve format used by `scripts/generate_figs.py`:

```bash
mv <run-dir> <run-dir>_legacy
python scripts/analyze/convert_prev_to_openevolve.py --src <run-dir>_legacy --dst <run-dir>
```

## Run PIKE-O

Clone and install [pike-openevolve](https://github.com/pike-project/pike-openevolve):

```bash
git clone git@github.com:pike-project/pike-openevolve.git
cd pike-openevolve
pip install -e .
```

Then run (from the pike-openevolve directory):

```bash
python examples/kernelbench/run.py --pike-dir <path/to/this-repo> --level 3-pike --task-start 1 --task-end 50 --max-fix-attempts 5 --eval-port 8000 --run-dir <path/to/output-dir>
```

To tune the PIKE-O configuration, edit `examples/kernelbench/config.yaml`.

## Running Multiple Searches Against One Eval Server

If you want to run several searches simultaneously (e.g., PIKE-B and PIKE-O in parallel), you can share a single Eval Worker but run separate Eval HTTP Servers on different ports, one per search process:

```bash
# Terminal 1: Eval Worker (shared)
python -u sandbox/tools/start_worker_container.py --engine docker --arch Hopper --max-active-tasks 20

# Terminal 2: Eval server for first search
python scripts/disk_channel_server.py --port 8000

# Terminal 3: Eval server for second search
python scripts/disk_channel_server.py --port 8001 --worker-io-dir worker_io_2

# Terminal 4: First search
python scripts/parallel_tree_search.py ... --eval-port 8000

# Terminal 5: Second search
python scripts/parallel_tree_search.py ... --eval-port 8001
```

When using `scripts/run_search.py`, pass `--no-eval-server` to skip starting an internal server (if you are managing it yourself), and use `--port` to point the search at the correct server.

## Remote Eval Server

The Eval HTTP Server is a plain HTTP server. In principle, the search process can connect to an Eval HTTP Server running on a different physical machine — for example, if the GPU node is separate from where the search process runs.

**This configuration has not been tested and requires caution.** The Eval HTTP Server has no authentication. You must ensure it is not exposed to the internet, e.g. by using SSH port forwarding:

```bash
# On the machine running the search process, forward a local port to the GPU node
ssh -L 8000:localhost:8000 <gpu-node-hostname>
```

Then run the Eval HTTP Server on the GPU node on port 8000 and point the search process at `localhost:8000`.
