# Add CLI Args to Fig/Table Generation Scripts
**Date:** 2026-03-04

## Goal
Refactor figure/table generation scripts to accept CLI arguments instead of hardcoded paths, and wire them into `generate_figs.py` to complete its TODO.

## Files Changed

### `scripts/analyze/merged_budget.py`
- Added argparse: `--input-dir`, `--output-dir`, `--level`, `--use-cost-stopping`, `--output-solutions`
- Core logic extracted into `run_level(input_dir, output_dir, level, use_cost_stopping, output_solutions)`; `main()` is a thin argparse wrapper around it
- Moved `runs` lists into a `runs_map` dict keyed by level (`"3-pike"` / `"5"`)
- Passed `run_name` explicitly to `get_progress_iters_attempts` (was a global)
- Replaced `curr_dir`-relative paths with `input_dir` / `output_dir` from args
- Config (`target_cost`, `results_dir`, `runtimes_dirname`, etc.) derived from args in `run_level()`

### `scripts/results/plot_trajectories.py`
- Added argparse: `--input-dir` (unused), `--output-dir`, `--level`
- `main(output_dir, level)` accepts parameters directly (argparse in `__main__` block)
- `target_dirname` now computed as `f"h100_level_{level}"`
- Conditionals changed from `target_dirname == "h100_level3-metr"` → `level == "3-pike"`
- Paths use `output_dir` instead of `curr_dir`

### `scripts/results/plot_overall_speedup.py`
- Added argparse: `--input-dir` (unused), `--output-dir`, `--level`
- `main(output_dir, level)` accepts parameters directly (argparse in `__main__` block)
- Moved module-level file loading and config into `main()`
- Baseline speedups (`metr`, `torch.compile`, `tensorrt`) moved into a `baseline_speedups_map` dict keyed by level
- Conditionals keyed on `level` instead of `target_dirname`

### `scripts/generate_figs.py`
- Imports `run_level` from `merged_budget`, `main` from `plot_trajectories` and `plot_overall_speedup`
- Calls these functions directly per level (no subprocess); replaced TODO comment with:
  - `merged_budget_run_level(...)` twice (with and without `use_cost_stopping=True`)
  - `plot_trajectories(output_dir, level)`
  - `plot_overall_speedup(output_dir, level)`

## Not Changed
- `scripts/results/gen_breakdown_table.py` — excluded per user request; retains hardcoded config
