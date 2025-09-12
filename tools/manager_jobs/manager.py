import os
import subprocess
from pathlib import Path

# Need to run: python scripts/disk_channel_server.py --port 8000

# Then need to run n of these in parallel, splitting up the task range:
# python examples/kernelbench/run.py --kernel_bench_dir /global/scratch/users/knagaitsev/KernelBench --level 3-metr --task_start 10 --task_end 20 --eval_port 8000

# Once all n of those are finished, need to:
# - kill disk_channel_server
# - send close message to the eval worker to end the H100 job (should we kill the slurm job instead?)

def main():
    port = 8000

    disk_channel_server = subprocess.Popen(["python", "scripts/disk_channel_server.py", "--port", str(port)])

    

    disk_channel_server.terminate()
    disk_channel_server.wait()

if __name__ == "__main__":
    main()
