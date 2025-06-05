import os
from pathlib import Path
import subprocess

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

def clean_whitespace(s):
    # split() splits string on any whitespace including \n and \t
    return ' '.join(s.split())

def main():
    # could be replaced with docker, the only flag that will need to be adjusted is --gpu (podman-hpc specific)
    container_cmd = "podman-hpc"

    kernel_bench_dir = Path.resolve(curr_path / "../..")

    run_cmd = ["bash"]

    # this command will be run in the container
    # eval_worker_path = "./scripts/start_eval_worker.py"
    # run_cmd = ["bash", "-c", f"pip install . && python3 {eval_worker_path}"]

    # TODO: can use $SLURM_PROCID env var for this, if it exists
    worker_id = str(0)

    # TODO: need to let this data_dir get passed in
    pscratch_dir = Path(os.getenv("PSCRATCH"))
    data_dir = pscratch_dir / "llm/KernelBench-data"

    worker_dir = data_dir / "workers" / worker_id
    input_dir = worker_dir / "input"
    output_dir = worker_dir / "output"

    cmd_str = f"""
            {container_cmd} run --gpu --cap-drop=ALL --network=none
            --tmpfs /cache
            --volume {kernel_bench_dir}:/app
            --volume {input_dir}:/input:ro
            --volume {output_dir}:/output
            --security-opt no-new-privileges --rm
            -it kernel-bench-deps
        """

    cleaned_cmd_str = clean_whitespace(cmd_str)

    cmd = cleaned_cmd_str.split()
    
    cmd += run_cmd

    print(f"Running: {cmd}")

    # TODO: may want to rebuild 'kernel-bench-deps' container beforehand

    proc = subprocess.Popen(
        cmd,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
        # env=env,
        # cwd=cwd,
    )
    proc.wait()

main()
