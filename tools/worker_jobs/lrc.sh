#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <port>"
    exit 1
fi

PORT=$1

IMAGE_PATH=container-images/kernel-bench-deps.sif

mkdir -p container-images
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Apptainer image does not exist, fetching it..."
    mkdir -p $SCRATCH/.cache/apptainer/tmp
    mkdir -p $SCRATCH/.cache/apptainer/cache

    # NOTE: must pass in tmp dir and cache dir, since otherwise /tmp does not have sufficient space
    # to set up the container on Lawrencium
    APPTAINER_TMPDIR=$SCRATCH/.cache/apptainer/tmp APPTAINER_CACHEDIR=$SCRATCH/.cache/apptainer/cache apptainer pull $IMAGE_PATH docker://docker.io/loonride/kernel-bench-deps:v0.4
fi

TIME_STR=$(date +"%Y-%m-%d-%H-%M-%S")

WORKER_IO_DIR=worker_io/workers/worker_$TIME_STR

mkdir -p $WORKER_IO_DIR

python scripts/disk_channel_server.py --port $PORT --worker_io_dir $WORKER_IO_DIR > /dev/null 2>&1 &
pid=$!
echo $pid > $WORKER_IO_DIR/disk_channel_server.pid

cleanup() {
    kill $pid 2>/dev/null
    rm -rf "$WORKER_IO_DIR"
}
trap cleanup EXIT

srun -A ac_binocular -t 24:00:00 --partition=es1 --qos=es_normal --gres=gpu:H100:4 --cpus-per-task=112 --pty python -u sandbox/tools/start_worker_container.py --engine apptainer --sif_path $IMAGE_PATH --worker_io_dir $WORKER_IO_DIR --arch Hopper
