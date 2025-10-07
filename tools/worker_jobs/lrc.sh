#!/bin/bash

IMAGE_PATH=container-images/kernel-bench-deps-v0.5.sif

mkdir -p container-images
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Apptainer image does not exist, fetching it..."
    mkdir -p $SCRATCH/.cache/apptainer/tmp
    mkdir -p $SCRATCH/.cache/apptainer/cache

    # NOTE: must pass in tmp dir and cache dir, since otherwise /tmp does not have sufficient space
    # to set up the container on Lawrencium
    APPTAINER_TMPDIR=$SCRATCH/.cache/apptainer/tmp APPTAINER_CACHEDIR=$SCRATCH/.cache/apptainer/cache apptainer pull $IMAGE_PATH docker://docker.io/loonride/kernel-bench-deps:v0.5
fi

# TIME_STR=$(date +"%Y-%m-%d-%H-%M-%S")

# # WORKER_IO_DIR=worker_io/workers/worker_$TIME_STR
# WORKER_IO_DIR=worker_io

# mkdir -p $WORKER_IO_DIR

# GPU_COUNT=8
# CPU_COUNT=112
# MAX_ACTIVE_TASKS=56

# srun -A ac_binocular -t 72:00:00 --partition=es1 --qos=es_normal --gres=gpu:H100:$GPU_COUNT --cpus-per-task=$CPU_COUNT --pty python -u sandbox/tools/start_worker_container.py --engine apptainer --sif_path $IMAGE_PATH --worker_io_dir $WORKER_IO_DIR --arch Hopper --max_active_tasks $MAX_ACTIVE_TASKS
