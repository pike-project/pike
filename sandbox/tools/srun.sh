# first run: alloc_gpu

# -n 1 -> this indicates how many instances of the container should run
# -G 4 -> this tells us how many GPUs are allocated per container

# (so if we wanted to deploy 8 workers, each with 4 GPUs, we do -n 8 -G 4)

# srun -n 4 -G 1 bash -c 'echo $SLURM_PROCID'

srun -n 1 -G 4 bash -c podman-hpc run --gpu --cap-drop=ALL --network=none \
    --tmpfs /cache \
    --volume $PSCRATCH/llm/KernelBench:/app \
    --volume $PSCRATCH/llm/KernelBench-data/workers/0/input:/input:ro \
    --volume $PSCRATCH/llm/KernelBench-data/workers/0/output:/output \
    --security-opt no-new-privileges --rm \
    -it kernel-bench-deps bash
