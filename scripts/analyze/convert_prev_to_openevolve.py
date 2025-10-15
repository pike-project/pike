import os
import json
import shutil
from pathlib import Path
import argparse
from collections import defaultdict

def numeric_suffix(name: str, prefix: str) -> int:
    """
    Extracts an integer suffix from directory/file names like 'task12', 'agent_3', etc.
    This is used for sorting directories numerically rather than lexicographically.
    """
    try:
        return int(name.replace(prefix, "").replace("_", ""))
    except ValueError:
        raise Exception(f"Numeric suffix failed: name -> {name}, prefix -> {prefix}")

def convert(src_dir: Path, dst_dir: Path):
    """
    Converts the "prev" directory structure to the "openevolve" structure,
    mapping each agent to an iter.

    Args:
        src_dir (Path): The root directory of the "prev" structure, containing task folders.
        dst_dir (Path): The root directory where the new "openevolve" structure will be created.
    """
    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dst_dir}")
    print("-" * 30)

    # 1. Find and sort all 'task' directories in the source
    try:
        src_task_dirs = sorted(
            [d for d in src_dir.iterdir() if d.is_dir() and d.name.startswith("task")],
            key=lambda p: numeric_suffix(p.name, "task")
        )
    except FileNotFoundError:
        print(f"Error: Source directory '{src_dir}' not found.")
        return

    if not src_task_dirs:
        print(f"Warning: No 'task' directories found in '{src_dir}'.")
        return

    # 2. Process each task directory
    for src_task_path in src_task_dirs:
        task_name = src_task_path.name
        print(f"Processing {task_name}...")

        # Data structure to hold all step paths for each agent, keyed by agent number.
        # This preserves the chronological order by traversing phases and steps sequentially.
        # e.g., { 0: [step_path1, step_path2, ...], 1: [step_pathA, step_pathB, ...] }
        agent_to_steps_map = defaultdict(list)

        src_phases_root = src_task_path / "phases"
        if not src_phases_root.is_dir():
            print(f"  - No 'phases' directory found in {task_name}, skipping.")
            continue

        # 3. Collect all step paths and group them by agent number.
        # The traversal order (phase -> agent -> step) is crucial to maintain chronology.
        phase_paths = sorted(
            [d for d in src_phases_root.iterdir() if d.is_dir() and d.name.startswith("phase_")],
            key=lambda p: numeric_suffix(p.name, "phase_")
        )

        for phase_path in phase_paths:
            src_agents_root = phase_path / "agents"
            if not src_agents_root.is_dir():
                continue

            agent_paths = sorted(
                [d for d in src_agents_root.iterdir() if d.is_dir() and d.name.startswith("agent_")],
                key=lambda p: numeric_suffix(p.name, "agent_")
            )

            for agent_path in agent_paths:
                agent_num = numeric_suffix(agent_path.name, "agent_")
                step_paths = sorted(
                    [d for d in agent_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
                    key=lambda p: numeric_suffix(p.name, "step_")
                )
                # Add all step paths from this agent instance to its collection
                agent_to_steps_map[agent_num].extend(step_paths)

        if not agent_to_steps_map:
            print(f"  - No agents or steps found for {task_name}, skipping.")
            continue

        # 4. Create the openevolve structure from the collected data.
        # Each agent becomes an iter, and its collected steps become attempts.
        total_attempts_converted = 0
        sorted_agent_nums = sorted(agent_to_steps_map.keys())

        for agent_num in sorted_agent_nums:
            iter_num = agent_num  # Direct mapping from agent number to iter number
            step_paths_for_this_iter = agent_to_steps_map[agent_num]

            if not step_paths_for_this_iter:
                continue  # Skip agents that existed but had no steps

            # Create the destination directory for this iter's attempts
            dst_attempts_root = dst_dir / task_name / "output" / "iter_output" / f"iter_{iter_num}" / "attempts"
            dst_attempts_root.mkdir(parents=True, exist_ok=True)

            # Each step path becomes a sequential attempt within this iter
            for attempt_num, step_path in enumerate(step_paths_for_this_iter):
                dst_attempt_path = dst_attempts_root / f"attempt_{attempt_num}"
                dst_attempt_path.mkdir(exist_ok=True)

                # --- File Conversion ---
                src_kernel_file = step_path / "kernel.py"
                src_eval_file = step_path / "eval_results.json"
                
                dst_code_file = dst_attempt_path / "code.py"
                dst_metrics_file = dst_attempt_path / "metrics_artifacts.json"

                # a) Copy kernel.py to code.py
                if src_kernel_file.exists():
                    shutil.copy(src_kernel_file, dst_code_file)

                # b) Transform eval_results.json to metrics_artifacts.json
                if src_eval_file.exists():
                    try:
                        with src_eval_file.open("r") as f:
                            eval_data = json.load(f)
                        runtime = eval_data.get("eval_results", {}).get("runtime")
                        if runtime is not None:
                            metrics_data = {"metrics": {"runtime": runtime}}
                            with dst_metrics_file.open("w") as f:
                                json.dump(metrics_data, f, indent=4)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"    - Warning: Could not process {src_eval_file}. Error: {e}")
                
                total_attempts_converted += 1

        print(f"  - Converted {total_attempts_converted} total attempts into {len(sorted_agent_nums)} iters for {task_name}.")

    print("\nConversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert a "prev" output directory structure to the "openevolve" structure.'
    )
    parser.add_argument("--src", type=str, required=True, help="Path to the source directory with the 'prev' structure.")
    parser.add_argument("--dst", type=str, required=True, help="Path to the destination directory for the 'openevolve' structure.")
    args = parser.parse_args()

    src_dir = Path(args.src) / "levels"
    src_dir = src_dir / (os.listdir(src_dir)[0])
    dst_dir = Path(args.dst) / "tasks"

    # Ensure the top-level destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    convert(src_dir, dst_dir)
