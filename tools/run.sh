set -x

DATA_DIR=$PSCRATCH/llm/KernelBench-data
RUN_NAME=$(date +"%Y_%m_%d_%H_%M_%S")
RUN_DIR=$DATA_DIR/runs/$RUN_NAME

mkdir -p $RUN_DIR
LOG_PATH=$RUN_DIR/out.log

KERNEL_BENCH_RUN_NAME=$RUN_NAME python scripts/parallel_tree_search.py data_dir=$DATA_DIR dataset_src=local server_type=google model_name=gemini-2.5-pro num_workers=50 worker_input_dir=worker_io/input worker_output_dir=worker_io/output level=0 task_start=1 task_end=1 num_samples=10 dry_run=True | tee $LOG_PATH
