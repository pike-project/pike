import os
from pathlib import Path
import subprocess
import argparse
import shutil

curr_path = Path(os.path.realpath(os.path.dirname(__file__)))

def clean_whitespace(s):
    # split() splits string on any whitespace including \n and \t
    return ' '.join(s.split())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine", type=str, required=False, default="docker", help="Container engine to use.")
    parser.add_argument("--sif_path", type=str, required=False, help="Path to the .sif file, required for Apptainer.")
    parser.add_argument("--arch", type=str, required=True, help="NVIDIA GPU Architecture")
    parser.add_argument("--bash", action='store_true', help="Run bash inside of the container instead of the worker")
    parser.add_argument("--pull_image", action='store_true', help="Pull the image from Dockerhub if using docker or podman-hpc")

    args = parser.parse_args()

    valid_engines = [
        "docker",
        "podman-hpc",
        "apptainer"
    ]

    if args.engine not in valid_engines:
        raise Exception(f"Invalid engine provided: {args.engine}, Valid engines: {valid_engines}")

    if args.engine == "apptainer" and args.sif_path is None:
        raise Exception("--sif_path argument must be provided if engine is 'apptainer'")
    
    valid_archs = [
        "Ampere", # A100
        "Hopper" # H100
    ]

    if args.arch not in valid_archs:
        raise Exception(f"Invalid arch: {args.arch}, Valid archs: {valid_archs}")
    
    sif_path = Path(args.sif_path) if args.sif_path else None

    # could be replaced with docker, the only flag that will need to be adjusted is --gpu (podman-hpc specific)
    container_cmd = args.engine

    local_image_name = "kernel-bench-deps"
    remote_image_name = "docker.io/loonride/kernel-bench-deps:v0.3"

    non_root_user = False
    read_only_fs = True
    pull_from_docker_hub = args.pull_image

    root_dir = Path.resolve(curr_path / "../..")

    app_bind_dir = Path("/app")

    # IMPORTANT: this command will be run in the container, so it is relative to the app bind dir
    if args.bash:
        run_cmd = ["bash"]
    else:
        eval_worker_path = app_bind_dir / "scripts/start_eval_worker.py"
        run_cmd = ["python", str(eval_worker_path), "--arch", args.arch]

    # TODO: can use $SLURM_PROCID env var for this, if it exists
    worker_id = str(0)

    # worker_dir = data_dir / "workers" / worker_id
    worker_dir = root_dir / "worker_io"

    input_dir = worker_dir / "input"
    output_dir = worker_dir / "output"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = worker_dir / "tmp"
    cache_dir = worker_dir / "cache"
    scratch_dir = worker_dir / "scratch"

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    if args.engine == "apptainer":
        # Apptainer command construction
        # Use 'exec' to run a custom command inside the container
        cmd = [container_cmd, "exec"]
        
        flags = [
            # --nv is the apptainer equivalent of --gpus for nvidia gpus
            "--nv",
            # --containall is a powerful isolation flag. It disables networking,
            # mounts a new /tmp, and uses a minimal /dev, which is a good
            # equivalent for docker's --network=none and --cap-drop=ALL.
            "--containall",
            # --no-privs is equivalent to --security-opt no-new-privileges
            "--no-privs",
            # --scratch creates a temporary directory inside the container,
            # equivalent to --tmpfs
            "--bind", f"{tmp_dir}:/tmp",
            "--bind", f"{cache_dir}:/cache",
            "--bind", f"{scratch_dir}:/scratch",
            # --bind is the apptainer equivalent of --volume
            "--bind", f"{input_dir}:/input",
            "--bind", f"{output_dir}:/output",
        ]

        # Apptainer runs as the current user by default, so the logic for
        # non_root_user and --userns keep-id is not needed.

        if read_only_fs:
            # :ro makes the bind mount read-only
            flags += ["--bind", f"{root_dir}:{app_bind_dir}:ro"]
        else:
            flags += ["--bind", f"{root_dir}:{app_bind_dir}"]
        
        cmd += flags

        # Add the SIF file path and the command to run
        cmd += [str(sif_path)]
        cmd += run_cmd

    else: # docker or podman-hpc
        flags_str = f"""
                --gpus all --cap-drop=ALL --network=none
                --tmpfs /tmp
                --tmpfs /cache
                --tmpfs /scratch
                --volume {input_dir}:/input
                --volume {output_dir}:/output
                --security-opt no-new-privileges --rm
                -it
            """

        cleaned_flags_str = clean_whitespace(flags_str)
        # For podman-hpc, replace '--gpus all' with its specific '--gpu' flag
        if args.engine == "podman-hpc":
            cleaned_flags_str = cleaned_flags_str.replace("--gpus all", "--gpu")
            
        flags = cleaned_flags_str.split()

        if non_root_user:
            flags += ["--userns", "keep-id"]

        if read_only_fs:
            # makes /app mount read-only as well
            flags += ["--read-only", "--volume", f"{root_dir}:{app_bind_dir}:ro"]
        else:
            # does not make /app mount read-only
            flags += ["--volume", f"{root_dir}:{app_bind_dir}"]

        cmd = [container_cmd, "run"]
        cmd += flags

        if pull_from_docker_hub:
            cmd += ["--pull=always", remote_image_name]
        else:
            cmd += [local_image_name]

        cmd += run_cmd

    print(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
        # env=env,
        # cwd=cwd,
    )
    proc.wait()

if __name__ == "__main__":
    main()
