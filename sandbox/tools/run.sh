podman-hpc run --gpu --cap-drop=ALL --network=none --userns user \
    --read-only \
    --volume $PSCRATCH/llm/KernelBench-data:/data \
    --security-opt no-new-privileges --rm \
    python-docker-app
    # -it ubuntu bash

# -d -> if we want to run it as a daemon


# --read-only -- this makes everything besides the mounted volume read only

# --tmpfs /workspace \

# --memory="256m" --memory-swap="256m" --cpus="0.5" --pids-limit=100 \

# --volume /path/to/input:/input:ro \ # Mount input data read-only
# --volume /path/to/output:/output \  # Mount output directory (be VERY cautious with write access)

# --security-opt seccomp=default
