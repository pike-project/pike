OUTPUT_DIR=data/runs/level_5_trial_1

python scripts/solution_eval/eval_solutions.py --level 5 --solutions baseline --mode eager --output_dir $OUTPUT_DIR --output_name eager --sequential
python scripts/solution_eval/eval_solutions.py --level 5 --solutions baseline --mode compile --output_dir $OUTPUT_DIR --output_name compile --sequential
python scripts/solution_eval/eval_solutions.py --level 5 --solutions baseline --mode tensorrt --output_dir $OUTPUT_DIR --output_name tensorrt --sequential
python scripts/solution_eval/eval_solutions.py --level 5 --solutions metr --mode eager --output_dir $OUTPUT_DIR --output_name metr --sequential
