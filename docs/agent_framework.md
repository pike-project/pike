# Agent Framework

An example of running the parallel tree search is shown below:

```bash
python -u scripts/parallel_tree_search.py data_dir=./data server_type=google model_name=gemini-2.5-pro num_workers=30 worker_input_dir=./worker_io/input worker_output_dir=./worker_io/output level=0 task_start=1 task_end=5 num_samples=10 num_phases=5 max_fix_attempts=5 dry_run=True
```

The following parameters are relevant:

- `server` and `model_name` - which LLM model to use
    - currently supported servers are `openai`, `google`, `cborg`
- `num_workers` - this is the number of workers for running LLM API queries in parallel, NOT related to the eval worker
- `level` - the KernelBench level to do (does not need to be a number, can also be a string like `3-metr`)
- `task_start` and `task_end` - the range of tasks in the level to perform, inclusive on both ends
- `num_samples` - How many agents to run in parallel **for each task**
- `num_phases` - Number of search phases to complete
- `max_fix_attempts` - Max fix attempts for the error fixing agent, before giving up and proceeding to next round
- `dry_run` - set to `True` if you want to test the agent framework without requiring the eval worker

## Prompt Construction

Prompt construction is done in `src/prompt_construction.py`

There are a few key prompts that are used by the agent:

- ideas prompt: prompts an LLM to brainstorm tasks that should be assigned to the performance engineering agents
- phase 0 performance optimization prompt: gives a performance engineering agent an example of what a model may look like after applying optimizations, then prompts the agent to optimize a specific pytorch model, seeding with an idea from the previous brainstorming agent
- phase 1-n performance optimization prompt: similar to the previous, but instead of giving a brainstorming idea, it gives a previous, well-performing solution, and prompts the agent to improve upon this previous solution
- error fixing prompt: tells the error fixing agent to fix a compilation/correctness issue in the model code that was produced by a previous performance optimization agent

## Query Strategies

Query strategies to drive the agent framework search are implemented in: `src/query_strategies.py`

TODO: more notes about query strategies
