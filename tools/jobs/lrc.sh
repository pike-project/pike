#!/bin/bash

mkdir -p container-images
if [ ! -f "" ]; then
    echo "Apptainer image does not exist, fetching it..."
    mkdir -p $SCRATCH/.cache/apptainer/tmp
    mkdir -p $SCRATCH/.cache/apptainer/cache

    # NOTE: must pass in tmp dir and cache dir, since otherwise /tmp does not have sufficient space
    # to set up the container on Lawrencium
    APPTAINER_TMPDIR=$SCRATCH/.cache/apptainer/tmp APPTAINER_CACHEDIR=$SCRATCH/.cache/apptainer/cache apptainer pull kernel-bench-deps.sif docker://docker.io/loonride/kernel-bench-deps:v0.3
fi

srun -A ac_binocular -t 24:00:00 --partition=es1 --qos=es_normal --gres=gpu:H100:4 --cpus-per-task=64 --pty python -u sandbox/tools/start_worker_container.py --engine apptainer --sif_path container-images/kernel-bench-deps.sif --arch Hopper
