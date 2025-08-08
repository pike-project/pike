# Agent Framework

An example of running the parallel tree search is shown below:

```bash
python -u scripts/parallel_tree_search.py data_dir=./data server_type=google model_name=gemini-2.5-pro num_workers=30 worker_input_dir=./worker_io/input worker_output_dir=./worker_io/output level=0 task_start=1 task_end=5 num_samples=10 num_phases=5 max_fix_attempts=5 dry_run=True
```

The following parameters are relevant:

- `server` and `model_name` - which LLM model to use
- `num_workers` - this is the number of workers for running LLM API queries in parallel, NOT related to the eval worker
- `level` - the KernelBench level to do (does not need to be a number, can also be a string like `3-metr`)
- `task_start` and `task_end` - the range of tasks in the level to perform, inclusive on both ends
- `num_samples` - How many agents to run in parallel **for each task**
- `num_phases` - Number of search phases to complete
- `max_fix_attempts` - Max fix attempts for the error fixing agent, before giving up and proceeding to next round
- `dry_run` - set to `True` if you want to test the agent framework without requiring the eval worker

## Prompt Construction

Prompt construction is done in `src/prompt_construction.py`

TODO: more notes about prompt construction

## Query Strategies

Query strategies to drive the agent framework search are implemented in: `src/query_strategies.py`

TODO: more notes about query strategies
