import os
from pathlib import Path
import subprocess
import argparse

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

def clean_whitespace(s):
    # split() splits string on any whitespace including \n and \t
    return ' '.join(s.split())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine", type=str, required=False, default="docker")
    parser.add_argument("--sif_path", type=str, required=False)
    args = parser.parse_args()

    valid_engines = [
        "docker",
        "podman-hpc",
        "apptainer"
    ]

    if args.engine not in valid_engines:
        raise Exception(f"Invalid engine provided: {args.engine}, Valid engines: {valid_engines}")

    if args.engine == "apptainer" and args.sif_path is None:
        raise Exception("sif_path argument must be provided if engine is Apptainer")
    
    sif_path = args.sif_path

    # could be replaced with docker, the only flag that will need to be adjusted is --gpu (podman-hpc specific)
    container_cmd = args.engine

    local_image_name = "kernel-bench-deps"
    remote_image_name = "docker.io/loonride/kernel-bench-deps:v0.3"

    non_root_user = False
    read_only_fs = True
    pull_from_docker_hub = False

    root_dir = Path.resolve(curr_path / "../..")

    # this command will be run in the container
    # IMPORTANT: DO NOT MAKE IT AN ABSOLUTE PATH, KEEP IT RELATIVE
    eval_worker_path = "./scripts/start_eval_worker.py"
    # run_cmd = ["bash", "-c", f"pip install . && python3 {eval_worker_path}"]
    run_cmd = ["python3", eval_worker_path]

    # run_cmd = ["bash"]

    # TODO: can use $SLURM_PROCID env var for this, if it exists
    worker_id = str(0)

    # worker_dir = data_dir / "workers" / worker_id
    worker_dir = root_dir / "worker_io"

    input_dir = worker_dir / "input"
    output_dir = worker_dir / "output"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    flags_str = f"""
            --gpu --cap-drop=ALL --network=none
            --tmpfs /cache
            --tmpfs /scratch
            --volume {input_dir}:/input
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
        flags += ["--read-only", "--volume", f"{root_dir}:/app:ro"]
    else:
        # does not make /app mount read-only
        flags += ["--volume", f"{root_dir}:/app"]

    cmd = [container_cmd, "run"]
    cmd += flags

    if pull_from_docker_hub:
        cmd += ["--pull=always", remote_image_name]
    else:
        cmd += [local_image_name]

    cmd += run_cmd

    print(f"Running: {" ".join(cmd)}")

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
