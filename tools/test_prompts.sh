set -ex

LEVEL=0
# inclusive range
TASK_START=1
TASK_END=1

NUM_SAMPLES=10
NUM_PHASES=3
MAX_FIX_ATTEMPTS=3

# True or False
DRY_RUN=False

DATA_DIR=$PSCRATCH/llm/KernelBench-data
RUN_NAME=$(date +"%Y_%m_%d_%H_%M_%S")
RUN_DIR=$DATA_DIR/runs/$RUN_NAME

mkdir -p $RUN_DIR
LOG_PATH=$RUN_DIR/out.log

WORKER_INPUT_DIR=worker_io/input
WORKER_OUTPUT_DIR=worker_io/output

python scripts/parallel_tree_search.py data_dir=$DATA_DIR run_dir=$RUN_DIR dataset_src=local server_type=cborg model_name=lbl/llama num_workers=50 worker_input_dir=$WORKER_INPUT_DIR worker_output_dir=$WORKER_OUTPUT_DIR level=$LEVEL task_start=$TASK_START task_end=$TASK_END num_samples=$NUM_SAMPLES num_phases=$NUM_PHASES max_fix_attempts=$MAX_FIX_ATTEMPTS dry_run=$DRY_RUN | tee -a $LOG_PATH

# python scripts/solution_eval/eval_solutions.py --level $LEVEL --mode eager --run_dir $RUN_DIR --worker_input_dir $WORKER_INPUT_DIR --worker_output_dir $WORKER_OUTPUT_DIR | tee -a $LOG_PATH
# python scripts/solution_eval/eval_solutions.py --level $LEVEL --mode compile --run_dir $RUN_DIR --worker_input_dir $WORKER_INPUT_DIR --worker_output_dir $WORKER_OUTPUT_DIR | tee -a $LOG_PATH

# python scripts/analyze/plot_phase_perf_improvement.py --run_dir $RUN_DIR | tee -a $LOG_PATH
