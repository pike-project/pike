set -ex

LEVEL=0

DATA_DIR=$PSCRATCH/llm/KernelBench-data
RUN_NAME=$(date +"%Y_%m_%d_%H_%M_%S")
RUN_DIR=$DATA_DIR/runs/$RUN_NAME

mkdir -p $RUN_DIR
LOG_PATH=$RUN_DIR/out.log

WORKER_INPUT_DIR=worker_io/input
WORKER_OUTPUT_DIR=worker_io/output

python scripts/parallel_tree_search.py data_dir=$DATA_DIR run_dir=$RUN_DIR dataset_src=local server_type=google model_name=gemini-2.5-pro num_workers=50 worker_input_dir=$WORKER_INPUT_DIR worker_output_dir=$WORKER_OUTPUT_DIR level=$LEVEL task_start=1 task_end=1 num_samples=10 dry_run=True | tee $LOG_PATH

python scripts/solution_eval/eval_solutions.py --level $LEVEL --mode eager --run_dir $RUN_DIR --worker_input_dir $WORKER_INPUT_DIR --worker_output_dir $WORKER_OUTPUT_DIR | tee $LOG_PATH
python scripts/solution_eval/eval_solutions.py --level $LEVEL --mode compile --run_dir $RUN_DIR --worker_input_dir $WORKER_INPUT_DIR --worker_output_dir $WORKER_OUTPUT_DIR | tee $LOG_PATH
