# Evaluation Worker

The recommended way to run the eval worker is in a sandbox container like so:

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper>
```

However, if you wish to run outside a sandbox container (not recommended), you can do so like this:

```bash
mkdir -p worker_io/input && mkdir -p worker_io/output && mkdir -p worker_io/scratch
python scripts/start_eval_worker.py --input-dir worker_io/input --output-dir worker_io/output --scratch-dir worker_io/scratch --arch <Ampere/Hopper>
```

## Design

The evaluation worker is placed in a container sandbox. It communicates with eval drivers and the agent framework via a filesystem I/O channel (NFS on HPC systems). The reason for this is that it is a simpler design for communication than using HTTP or WebSockets (though these could be implemented via the channel interface as well, if needed). It is also safer, since we do not need to make any network available to the container when using this approach, we just have to mount certain directories as volumes in the container.

The eval worker uses async Python API to await new tasks on the channel, then spins up an async eval task when a new task arrives. The async eval task runs a `scripts/eval.py` subprocess, telling it which code to run. the `eval.py` script is what runs the LLM-generated code. If something goes wrong, the main eval worker can kill the eval task after a set timeout.

The number of concurrent tasks is limited to a reasonable number, usually 20 for a 64+ thread CPU. The reasoning behind this is that each CUDA compilation typically usese 1 thread. Pytorch model performance runs are happening simultaneously. While these performance runs are GPU-heavy workloads, they may have to synchronize with the CPU at times, so we do not want the CPUs to be swamped with other tasks.

To ensure peak GPU performance, each available GPU is given a lockfile. For a given `eval.py` run to use a GPU, it must first claim exclusive access of this lockfile. This ensures no two tasks are being evaluated on a given GPU at the same time.
