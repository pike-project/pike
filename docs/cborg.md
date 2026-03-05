## CBORG Instructions

Save the following API key environment variable to `~/.bashrc`:

```bash
export CBORG_API_KEY=<...>
```

Then source the changes via:

```bash
source ~/.bashrc
```

First, start the Eval Worker as described in the top-level README.

Then run a search with a free CBORG model using `scripts/run_search.py` (which manages the eval HTTP server automatically):

```bash
python scripts/run_search.py \
    --output-dir data/pike-data \
    --strategy pike-b \
    --level 3-pike \
    --server-type cborg \
    --model-name lbl/gpt-oss-120b-high \
    --run-name h100_level_3-pike_pike-b \
    --task-start 1 --task-end 50
```

If you prefer to run components separately (e.g. to manage the eval HTTP server yourself), see [`advanced_setup.md`](advanced_setup.md). In that case, pass `--server-type cborg --model-name lbl/gpt-oss-120b-high` to `scripts/parallel_tree_search.py`.
