# Add CLI Args to Fig/Table Generation Scripts
**Date:** 2026-03-04

## Goal
Refactor figure/table generation scripts to accept CLI arguments instead of hardcoded paths, and wire them into `generate_figs.py` to complete its TODO.

## Files Changed

### `scripts/analyze/merged_budget.py`
- Added argparse: `--input-dir`, `--output-dir`, `--level`, `--use-cost-stopping`, `--output-solutions`
- Wrapped all execution in `main()`
- Moved `runs` lists into a `runs_map` dict keyed by level (`"3-pike"` / `"5"`)
- Passed `run_name` explicitly to `get_progress_iters_attempts` (was a global)
- Replaced `curr_dir`-relative paths with `input_dir` / `output_dir` from args
- Config (`target_cost`, `results_dir`, `runtimes_dirname`, etc.) derived from args in `main()`

### `scripts/results/plot_trajectories.py`
- Added argparse: `--input-dir` (unused), `--output-dir`, `--level`
- `target_dirname` now computed as `f"h100_level_{level}"`
- Conditionals changed from `target_dirname == "h100_level3-metr"` → `level == "3-pike"`
- Paths use `output_dir` instead of `curr_dir`

### `scripts/results/plot_overall_speedup.py`
- Added argparse: `--input-dir` (unused), `--output-dir`, `--level`
- Moved module-level file loading and config into `main(output_dir, level)`
- Baseline speedups (`metr`, `torch.compile`, `tensorrt`) moved into a `baseline_speedups_map` dict keyed by level
- Conditionals keyed on `level` instead of `target_dirname`

### `scripts/generate_figs.py`
- Added `import subprocess, sys`
- Replaced TODO comment with subprocess calls per level:
  - `merged_budget.py` twice (with and without `--use-cost-stopping`)
  - `plot_trajectories.py`
  - `plot_overall_speedup.py`

## Not Changed
- `scripts/results/gen_breakdown_table.py` — excluded per user request; retains hardcoded config
