"""
Collect all baseline runtimes (eager, compile, tensorrt, metr) and write them
to the pike-data directory structure.

Usage:
    python scripts/eval_baselines.py \
        --output-dir data/pike-data \
        --level 3-pike \
        [--port 8000] \
        [--worker-io-dir worker_io] \
        [--dry-run]
"""

import os
import sys
import asyncio
import argparse
import subprocess
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.solution_eval.eval_solutions import EvalSolutions

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
root_dir = (curr_dir / "..").resolve()
deps_dir = root_dir / "local/deps"


def fetch_metr_deps():
    """Clone KernelBenchFiltered into local/deps/ if not present."""
    deps_dir.mkdir(parents=True, exist_ok=True)
    kb_filtered = deps_dir / "KernelBenchFiltered"
    if not kb_filtered.exists():
        print("Cloning KernelBenchFiltered...")
        subprocess.run(
            ["git", "clone", "git@github.com:METR/KernelBenchFiltered.git"],
            check=True,
            cwd=deps_dir,
        )
    else:
        print("KernelBenchFiltered already present, skipping clone.")


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


async def run_baseline(
    solutions: str,
    mode: str,
    output_name: str,
    output_dir: Path,
    level: str,
    eval_port: int,
    dry_run: bool,
):
    """Run one baseline evaluation and write results JSON."""
    mode_title_map = {
        "eager": "Eager",
        "compile": "torch.compile",
        "tensorrt": "TensorRT",
    }
    solutions_title_map = {
        "baseline": "",
        "metr": "METR",
    }

    if solutions == "baseline":
        title = mode_title_map[mode]
    else:
        title = solutions_title_map[solutions]

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_sol = EvalSolutions(
        level=level,
        mode=mode,
        solutions_name=solutions,
        title=title,
        run_dir=None,
        output_name=output_name,
        output_dir=output_dir,
        eval_port=eval_port,
        dry_run=dry_run,
        sequential=True,
    )
    await eval_sol.run()
    print(f"Wrote {output_dir / output_name}.json")


async def run_all_baselines(args):
    output_dir = Path(args.output_dir)
    level = args.level
    worker_io_dir = Path(args.worker_io_dir)
    dry_run = args.dry_run

    baseline_dir = output_dir / "baseline-runtimes" / f"h100_level_{level}"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        fetch_metr_deps()

    server_proc = None
    if not args.no_eval_server and not dry_run:
        server_proc = start_eval_server(args.port, worker_io_dir)

    try:
        evals = [
            # ("baseline", "eager",     "eager"),
            ("baseline", "compile",   "compile"),
            ("baseline", "tensorrt",  "tensorrt"),
            ("metr",     "eager",     "metr"),
        ]

        for solutions, mode, output_name in evals:
            print(f"\n--- Running {output_name} baseline ---")
            await run_baseline(
                solutions=solutions,
                mode=mode,
                output_name=output_name,
                output_dir=baseline_dir,
                level=level,
                eval_port=args.port,
                dry_run=dry_run,
            )
    finally:
        if server_proc is not None:
            print("Stopping eval HTTP server...")
            server_proc.terminate()
            server_proc.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Collect all baseline runtimes and write to pike-data directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root pike-data directory (e.g. data/pike-data)",
    )
    parser.add_argument(
        "--level", type=str, required=True,
        help="KernelBench level (e.g. 3-pike, 5)",
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
        help="Dry run: produce dummy results without hitting the eval worker",
    )
    args = parser.parse_args()

    asyncio.run(run_all_baselines(args))


if __name__ == "__main__":
    main()
