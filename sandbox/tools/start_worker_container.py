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
    image_name = "kernel-bench-deps"

    non_root_user = True
    read_only_fs = True

    kernel_bench_dir = Path.resolve(curr_path / "../..")

    # this command will be run in the container
    eval_worker_path = "./scripts/start_eval_worker.py"
    # run_cmd = ["bash", "-c", f"pip install . && python3 {eval_worker_path}"]
    run_cmd = ["python3", eval_worker_path]

    # run_cmd = ["bash"]

    # TODO: can use $SLURM_PROCID env var for this, if it exists
    worker_id = str(0)

    # TODO: need to let this data_dir get passed in
    pscratch_dir = Path(os.getenv("PSCRATCH"))
    data_dir = pscratch_dir / "llm/KernelBench-data"

    worker_dir = data_dir / "workers" / worker_id
    input_dir = worker_dir / "input"
    output_dir = worker_dir / "output"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    flags_str = f"""
            --gpu --cap-drop=ALL --network=none
            --tmpfs /cache
            --tmpfs /scratch
            --volume {input_dir}:/input:ro
            --volume {output_dir}:/output
            --security-opt no-new-privileges --rm
            -it
        """

    cleaned_flags_str = clean_whitespace(flags_str)
    flags = cleaned_flags_str.split()

    if non_root_user:
        flags += ["--userns", "keep-id"]

    if read_only_fs:
        # makes /app mount read-only as well
        flags += ["--read-only", "--volume", f"{kernel_bench_dir}:/app:ro"]
    else:
        # does not make /app mount read-only
        flags += ["--volume", f"{kernel_bench_dir}:/app"]

    cmd = [container_cmd, "run"]
    cmd += flags
    cmd += [image_name]
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
