# AGENTS.md

## Repository Structure

### Figure/table generation (`scripts/`)
- `generate_figs.py` — top-level orchestrator; call with `--input-dir` and `--output-dir`; iterates over levels `3-pike` and `5`, copies baseline runtimes, then calls the scripts below
- `scripts/analyze/merged_budget.py` — processes raw run outputs into per-task best runtimes and speedup trajectory CSVs; entry point `run_level(input_dir, output_dir, level, ...)`; run twice per level (with/without `--use-cost-stopping`) to produce `speedups.json` and `speedups_money_budget.json`
- `scripts/results/plot_trajectories.py` — plots geomean speedup vs. attempts/cost curves as a side-by-side PDF; `main(output_dir, level)`
- `scripts/results/plot_overall_speedup.py` — bar chart of final geomean speedups across all runs; `main(output_dir, level)`
- `scripts/results/gen_breakdown_table.py` — per-task speedup breakdown plot and LaTeX table; currently has hardcoded paths/config (not wired into `generate_figs.py`)

Output lands under `<output-dir>/h100_level_{level}/results/{data,figs}/`.

## Planning

- Always check `agent-plans/` directory for current architecture plans (if it exists)
- If making a complex plan, create a new plan file, and include the date in the plan filename to make it clear which plans were the most recent

## Code style preferences

- Prefer clean, root-cause fixes over hacky patches. Don't just wrap things in try-catch without understanding why the error happens.
- When investigating bugs, explain the underlying mechanism before proposing a fix.

## Commit style

- Do not include "Co-Authored-By" or "written by Claude" tags in commits unless explicitly asked.
