#!/usr/bin/env python3
import argparse
import os
import sys
import signal
import subprocess
from datetime import datetime
from pathlib import Path

def main():
    # Base directory = location of this script
    curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
    root_dir = (curr_dir / "../..").resolve()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Launch worker container job")
    parser.add_argument(
        "--worker_io_dir",
        type=str,
        default=None,
        help="Directory for worker I/O (default: worker_io relative to script)"
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=8,
        help="Number of GPUs to request (default: 8)"
    )
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=112,
        help="Number of CPUs per task (default: 112)"
    )
    parser.add_argument(
        "--max_active_tasks",
        type=int,
        default=56,
        help="Maximum number of active tasks (default: 56)"
    )
    parser.add_argument(
        "--allocation_time",
        type=str,
        default="72:00:00",
        help="Allocation time in Slurm format (default: 72:00:00)"
    )
    args = parser.parse_args()

    # Paths
    image_dir = root_dir / "container-images"
    image_path = image_dir / "kernel-bench-deps.sif"

    # Ensure container images directory exists
    image_dir.mkdir(parents=True, exist_ok=True)

    # If container image doesn't exist, pull it
    if not image_path.exists():
        print("Apptainer image does not exist, fetching it...")

        scratch = Path(os.environ.get("SCRATCH", str(Path.home() / "scratch")))
        tmp_dir = scratch / ".cache/apptainer/tmp"
        cache_dir = scratch / ".cache/apptainer/cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["APPTAINER_TMPDIR"] = str(tmp_dir)
        env["APPTAINER_CACHEDIR"] = str(cache_dir)

        subprocess.run(
            [
                "apptainer", "pull",
                str(image_path),
                "docker://docker.io/loonride/kernel-bench-deps:v0.5"
            ],
            check=True,
            env=env
        )

    # Time-stamped string
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Worker I/O directory
    if args.worker_io_dir:
        worker_io_dir = Path(args.worker_io_dir)
    else:
        worker_io_dir = root_dir / "worker_io"  # Or timestamped if desired
        # worker_io_dir = root_dir / f"worker_io/workers/worker_{time_str}"

    worker_io_dir.mkdir(parents=True, exist_ok=True)

    # Build srun command
    cmd = [
        "srun",
        "-A", "ac_binocular",
        "-t", args.allocation_time,
        "--partition=es2",
        "--qos=es2_normal",
        f"--gres=gpu:H100:{args.gpu_count}",
        f"--cpus-per-task={args.cpu_count}",
        # "--pty",
        "python", "-u",
        str(root_dir / "sandbox/tools/start_worker_container.py"),
        "--engine", "apptainer",
        "--sif_path", str(image_path),
        "--worker_io_dir", str(worker_io_dir),
        "--arch", "Hopper",
        "--max_active_tasks", str(args.max_active_tasks),
    ]

    proc = subprocess.Popen(cmd)

    def handle_term(signum, frame):
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    proc.wait()

if __name__ == "__main__":
    main()
