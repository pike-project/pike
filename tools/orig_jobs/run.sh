set -ex

LEVEL=3-metr
# inclusive range
TASK_START=1
TASK_END=50

NUM_SAMPLES=10
NUM_PHASES=30
MAX_FIX_ATTEMPTS=5

# True or False
DRY_RUN=False

SERVER_TYPE=google
MODEL_NAME=gemini-2.5-pro
# SERVER_TYPE=cborg
# MODEL_NAME=lbl/llama

# IMPORTANT NOTE: this script currently makes a data directory outside of the current KernelBench
# directory, this needs to be modified if this behavior is not desired
DATA_DIR=$(realpath ./data)
RUN_NAME=$(date +"%Y_%m_%d_%H_%M_%S")
RUN_DIR=$DATA_DIR/runs/$RUN_NAME

mkdir -p $RUN_DIR
LOG_PATH=$RUN_DIR/out.log

WORKER_INPUT_DIR=worker_io/input
WORKER_OUTPUT_DIR=worker_io/output

# "python -u" makes output unbuffered, so we can see it immediately
python -u scripts/parallel_tree_search.py run_dir=$RUN_DIR server_type=$SERVER_TYPE model_name=$MODEL_NAME num_workers=30 worker_input_dir=$WORKER_INPUT_DIR worker_output_dir=$WORKER_OUTPUT_DIR level=$LEVEL task_start=$TASK_START task_end=$TASK_END num_samples=$NUM_SAMPLES num_phases=$NUM_PHASES max_fix_attempts=$MAX_FIX_ATTEMPTS dry_run=$DRY_RUN | tee -a $LOG_PATH

if [ "$DRY_RUN" = "True" ]; then
    DRY_RUN_FLAG="--dry_run"
else
    DRY_RUN_FLAG=""
fi

python -u scripts/solution_eval/eval_solutions.py --level $LEVEL --mode eager --output_name baseline_eager --output_dir $RUN_DIR --worker_input_dir $WORKER_INPUT_DIR --worker_output_dir $WORKER_OUTPUT_DIR $DRY_RUN_FLAG | tee -a $LOG_PATH
# IMPORTANT: the last eval_solutions call has the --close_worker flag to ensure the worker is closed on completion
python -u scripts/solution_eval/eval_solutions.py --level $LEVEL --mode compile --output_name baseline_compile --output_dir $RUN_DIR --worker_input_dir $WORKER_INPUT_DIR --worker_output_dir $WORKER_OUTPUT_DIR $DRY_RUN_FLAG --close_worker | tee -a $LOG_PATH

# python -u scripts/analyze/plot_phase_perf_improvement.py --run_dir $RUN_DIR | tee -a $LOG_PATH
