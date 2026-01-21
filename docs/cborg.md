## CBORG Instructions

Save the following API key environment variable to `~/.bashrc`:

```bash
export CBORG_API_KEY=<...>
```

Then source the changes via:

```bash
source ~/.bashrc
```

**First, set up the Eval Worker and Eval Server as described in the top-level README**

Now, try with a free CBORG model:

```bash
python scripts/parallel_tree_search.py --server-type cborg --model-name lbl/gpt-oss-120b-high --level 3-pike --task-start 1 --task-end 50 --num-branches 10 --max-fix-attempts 5 --query-budget 300 --eval-port 8000 --run-dir <path/to/output-dir>
```
