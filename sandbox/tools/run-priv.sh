#!/bin/bash

podman-hpc run --gpu \
    --tmpfs /cache \
    --volume $PSCRATCH/llm/KernelBench:/app \
    --volume $PSCRATCH/llm/KernelBench/data/workers/0/input:/input:ro \
    --volume $PSCRATCH/llm/KernelBench/data/workers/0/output:/output \
    --rm \
    -it kernel-bench-deps bash


# podman-hpc run --gpu --cap-drop=ALL --network=none \
#     --userns keep-id \
#     --read-only \
#     --tmpfs /cache \
#     --volume $PSCRATCH/llm/KernelBench:/app:ro \
#     --volume $PSCRATCH/llm/KernelBench/data/workers/0/input:/input:ro \
#     --volume $PSCRATCH/llm/KernelBench/data/workers/0/output:/output \
#     --security-opt no-new-privileges --rm \
#     -it kernel-bench-deps bash


# podman-hpc run --gpu --cap-drop=ALL --network=none \
#     --userns keep-id \
#     --read-only \
#     --tmpfs /cache \
#     --volume $PSCRATCH/llm/KernelBench/data:/data \
#     --security-opt no-new-privileges --rm \
#     -it python-docker-app bash
    # python-docker-app


# -d -> if we want to run it as a daemon


# --read-only -- this makes everything besides the mounted volume read only

# --tmpfs /workspace \

# --memory="256m" --memory-swap="256m" --cpus="0.5" --pids-limit=100 \

# --volume /path/to/input:/input:ro \ # Mount input data read-only
# --volume /path/to/output:/output \  # Mount output directory (be VERY cautious with write access)


# ideally should pass the following options with default Docker (not podman-hpc)
# --userns=default --user 1000:1000
# --security-opt seccomp=default
