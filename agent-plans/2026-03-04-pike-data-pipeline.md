# Plan: Streamline PIKE Data Collection Pipeline (2026-03-04)

## Overview

1. **Split and pin dependencies** — host vs eval-worker, exact versions
2. **New `scripts/eval_baselines.py`** — collects all baseline runtimes, runs eval HTTP server internally
3. **New `scripts/run_search.py`** — unified PIKE-B/PIKE-O entry point, runs eval HTTP server internally (optional)
4. **Update `sandbox/Dockerfile.deps`** — install both requirement files

---

## 1. Split and Pin Dependencies

### `requirements.txt` (host-only, no PyTorch/CUDA)

```
openai==2.15.0
anthropic==0.76.0
google-genai==1.58.0
together==1.5.34
aiofiles==25.1.0
requests==2.32.5
filelock==3.20.3
tqdm==4.67.1
numpy==2.4.1
packaging==25.0
transformers==4.57.5
matplotlib==3.10.8
pandas==2.3.3
scipy==1.17.0
```

**Removed**: `torch`, `torch_tensorrt`, `tensorrt`, `ninja`, `einops`, `datasets`, `modal`, `pydra_config`, `archon-ai`, `pytest`

### `requirements_eval_worker.txt` (eval container additional deps)

```
torch==2.8.0
torch_tensorrt==2.8.0
tensorrt==10.12.0.36
ninja==1.13.0
einops==0.8.1
datasets==4.5.0
diffusers
opt_einsum
flash_attn
```

### `sandbox/Dockerfile.deps` changes

- COPY and install `requirements.txt` + `requirements_eval_worker.txt`
- Remove `requirements_level5.txt` reference (merged into `requirements_eval_worker.txt`)

---

## 2. `scripts/eval_baselines.py`

```bash
python scripts/eval_baselines.py \
    --output-dir <pike-data-dir> \
    --level 3-pike \
    [--port 8000] \
    [--dry-run]
```

Behavior:
1. Clone `KernelBenchFiltered` to `local/deps/` via subprocess if not present (for METR)
2. Start eval HTTP server as background subprocess
3. Run 4 evals via `EvalSolutions` (reused from `scripts/solution_eval/eval_solutions.py`):
   - `baseline/eager` → `eager.json`
   - `baseline/compile` → `compile.json`
   - `baseline/tensorrt` → `tensorrt.json`
   - `metr/eager` → `metr.json`
4. Write to `<output-dir>/baseline-runtimes/h100_level_{level}/`

---

## 3. `scripts/run_search.py`

```bash
python scripts/run_search.py \
    --output-dir <pike-data-dir> \
    --strategy pike-b \
    --level 3-pike \
    --server-type google --model-name gemini-2.5-pro \
    --run-name h100_level_3-pike_pike-b \
    [--task-start 1] [--task-end 50] \
    [--query-budget 300] [--num-branches 10] [--max-fix-attempts 5] \
    [--port 8000] \
    [--no-eval-server] \
    [--dry-run]
```

Behavior:
1. Start eval HTTP server as background subprocess (unless `--no-eval-server`)
2. Pick next `run_N` number under `<output-dir>/full-pike-runs/level_<level>/<run-name>/runs/runs/`
3. PIKE-B: run `parallel_tree_search.py` via subprocess; then rename `run/` → `run_legacy/` and call `convert_prev_to_openevolve.py`
4. PIKE-O: clone `pike-openevolve` to `local/deps/` if needed, `pip install -e`, then run `examples/kernelbench/run.py` via subprocess

---

## 4. User Workflow

```bash
# Start eval worker (unchanged)
python -u sandbox/tools/start_worker_container.py --engine docker --arch Hopper --max-active-tasks 20

# Collect baselines (manages its own eval HTTP server)
python scripts/eval_baselines.py --output-dir data/pike-data --level 3-pike

# Run search (manages its own eval HTTP server, or pass --no-eval-server)
python scripts/run_search.py --output-dir data/pike-data --strategy pike-b \
    --level 3-pike --server-type google --model-name gemini-2.5-pro \
    --run-name h100_level_3-pike_pike-b

# Generate figures
python scripts/generate_figs.py --input-dir data/pike-data --output-dir output/figs
```
