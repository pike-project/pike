#!/bin/bash

rm -rf data/dry-run
python scripts/run_search.py --run-name test1 --output-dir data/dry-run/pike-data --strategy pike-b --level 3-pike --server-type google --model-name gemini-2.5-pro --task-start 1 --task-end 50 --dry-run
python scripts/eval_baselines.py --output-dir data/dry-run/pike-data --level 3-pike --dry-run
python scripts/generate_figs.py --input-dir data/dry-run/pike-data --output-dir data/dry-run/pike-out
