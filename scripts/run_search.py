"""
Unified entry point for running PIKE-B or PIKE-O search, with the eval HTTP
server started automatically.

Usage:
    python scripts/run_search.py \
        --output-dir data/pike-data \
        --strategy pike-b \
        --level 3-pike \
        --server-type google \
        --model-name gemini-2.5-pro \
        --run-name h100_level_3-pike_pike-b \
        [--task-start 1] [--task-end 50] \
        [--query-budget 300] \
        [--num-branches 10] \
        [--max-fix-attempts 5] \
        [--port 8000] \
        [--worker-io-dir worker_io] \
        [--no-eval-server] \
        [--dry-run]
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
root_dir = (curr_dir / "..").resolve()
deps_dir = root_dir / "local/deps"


def start_eval_server(port: int, worker_io_dir: Path) -> subprocess.Popen:
    """Launch the eval HTTP server as a background subprocess."""
    worker_io_dir.mkdir(parents=True, exist_ok=True)
    (worker_io_dir / "input").mkdir(parents=True, exist_ok=True)
    (worker_io_dir / "output").mkdir(parents=True, exist_ok=True)

    server_script = curr_dir / "disk_channel_server.py"
    proc = subprocess.Popen(
        [
            sys.executable, str(server_script),
            "--port", str(port),
            "--worker-io-dir", str(worker_io_dir),
        ],
        cwd=root_dir,
    )
    # Give the server a moment to start
    time.sleep(2)
    print(f"Started eval HTTP server on port {port} (pid={proc.pid})")
    return proc


def next_run_number(runs_dir: Path) -> int:
    """Find the next available run_N index under runs_dir."""
    existing = []
    if runs_dir.exists():
        for entry in runs_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("run_"):
                try:
                    existing.append(int(entry.name[len("run_"):]))
                except ValueError:
                    pass
    return max(existing, default=-1) + 1


def run_pike_b(args, run_dir: Path):
    """Run PIKE-B search via subprocess."""
    search_script = curr_dir / "parallel_tree_search.py"

    cmd = [
        sys.executable, str(search_script),
        "--run-dir", str(run_dir),
        "--server-type", args.server_type,
        "--model-name", args.model_name,
        "--level", args.level,
        "--task-start", str(args.task_start),
        "--task-end", str(args.task_end),
        "--num-branches", str(args.num_branches),
        "--max-fix-attempts", str(args.max_fix_attempts),
        "--eval-port", str(args.port),
        "--query-budget", str(args.query_budget),
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"Running PIKE-B: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=root_dir)


def convert_to_openevolve(run_dir: Path):
    """
    Convert PIKE-B legacy format to openevolve format.
    Renames run/ -> run_legacy/, then converts run_legacy/ -> run/.
    """
    legacy_dir = run_dir.parent / "run_legacy"
    openevolve_dir = run_dir.parent / "run"

    # run/ currently holds the legacy PIKE-B output; rename it
    run_dir.rename(legacy_dir)
    print(f"Renamed {run_dir} -> {legacy_dir}")

    convert_script = curr_dir / "analyze" / "convert_prev_to_openevolve.py"
    cmd = [
        sys.executable, str(convert_script),
        "--src", str(legacy_dir),
        "--dst", str(openevolve_dir),
    ]
    print(f"Converting to openevolve format: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=root_dir)


def run_pike_o(args, run_dir: Path):
    """Clone pike-openevolve if needed, install it, then run the search."""
    pike_oe_dir = deps_dir / "pike-openevolve"

    if not pike_oe_dir.exists():
        print("Cloning pike-openevolve...")
        deps_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "git@github.com:google-deepmind/pike-openevolve.git"],
            check=True,
            cwd=deps_dir,
        )

    print("Installing pike-openevolve...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        cwd=pike_oe_dir,
    )

    run_script = pike_oe_dir / "examples" / "kernelbench" / "run.py"
    cmd = [
        sys.executable, str(run_script),
        "--pike-dir", str(root_dir),
        "--output-dir", str(run_dir),
        "--level", args.level,
        "--server-type", args.server_type,
        "--model-name", args.model_name,
        "--task-start", str(args.task_start),
        "--task-end", str(args.task_end),
        "--eval-port", str(args.port),
    ]
    print(f"Running PIKE-O: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=root_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run PIKE-B or PIKE-O search and write results to pike-data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root pike-data directory (e.g. data/pike-data)",
    )
    parser.add_argument(
        "--strategy", type=str, required=True, choices=["pike-b", "pike-o"],
        help="Search strategy: pike-b or pike-o",
    )
    parser.add_argument(
        "--level", type=str, required=True,
        help="KernelBench level (e.g. 3-pike, 5)",
    )
    parser.add_argument(
        "--server-type", type=str, required=True,
        help="LLM server type (e.g. google, anthropic, openai)",
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        help="LLM model name (e.g. gemini-2.5-pro)",
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Name for this run (used as output subdirectory, e.g. h100_level_3-pike_pike-b)",
    )
    parser.add_argument(
        "--task-start", type=int, default=1,
        help="Task range start, inclusive (default: 1)",
    )
    parser.add_argument(
        "--task-end", type=int, default=50,
        help="Task range end, inclusive (default: 50)",
    )
    parser.add_argument(
        "--query-budget", type=int, default=300,
        help="Max LLM query budget (default: 300)",
    )
    parser.add_argument(
        "--num-branches", type=int, default=10,
        help="Number of parallel branches for PIKE-B (default: 10)",
    )
    parser.add_argument(
        "--max-fix-attempts", type=int, default=5,
        help="Max error-fix attempts per branch for PIKE-B (default: 5)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for the eval HTTP server (default: 8000)",
    )
    parser.add_argument(
        "--worker-io-dir", type=str, default="worker_io",
        help="Scratch directory for worker communication (default: worker_io)",
    )
    parser.add_argument(
        "--no-eval-server", action="store_true",
        help="Skip starting the eval HTTP server (use if already running)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run: skip LLM queries and eval worker communication",
    )
    args = parser.parse_args()

    if args.strategy == "pike-o" and args.dry_run:
        parser.error("--dry-run is not supported for pike-o")

    output_dir = Path(args.output_dir)
    worker_io_dir = Path(args.worker_io_dir)

    # Determine output location: <output-dir>/full-pike-runs/<run-name>/runs/runs/run_N/
    run_name_dir = output_dir / "full-pike-runs" / args.run_name / "runs" / "runs"
    run_n = next_run_number(run_name_dir)
    run_n_dir = run_name_dir / f"run_{run_n}"
    run_dir = run_n_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    server_proc = None
    if not args.no_eval_server:
        server_proc = start_eval_server(args.port, worker_io_dir)

    try:
        if args.strategy == "pike-b":
            run_pike_b(args, run_dir)
            convert_to_openevolve(run_dir)
        else:
            run_pike_o(args, run_dir)
    finally:
        if server_proc is not None:
            print("Stopping eval HTTP server...")
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
