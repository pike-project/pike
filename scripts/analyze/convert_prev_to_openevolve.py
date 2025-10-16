import os
import json
import shutil
from pathlib import Path
import argparse

def numeric_suffix(name: str, prefix: str) -> int:
    """
    Extracts an integer suffix from directory/file names like 'task12', 'agent_3', etc.
    This is used for sorting directories numerically rather than lexicographically.
    """
    try:
        return int(name.replace(prefix, "").replace("_", ""))
    except ValueError:
        raise Exception(f"Numeric suffix failed: name -> {name}, prefix -> {prefix}")

def copy_file(src_path, dst_path):
    if src_path.exists():
        shutil.copy(src_path, dst_path)

def convert(src_dir: Path, dst_dir: Path):
    """
    Converts the "prev" directory structure to the "openevolve" structure.
    Each unique agent directory encountered chronologically becomes a new 'iter'.

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
        task_num = int(task_name.split("_")[1])

        print(f"Processing {task_name}...")

        src_phases_root = src_task_path / "phases"
        if not src_phases_root.is_dir():
            print(f"  - No 'phases' directory found in {task_name}, skipping.")
            continue

        # This counter will increment for every agent directory we process, ensuring
        # a unique, sequential iter_id.
        iter_counter = 0
        total_attempts_converted = 0

        # 3. Traverse phases and agents in chronological order.
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

            # Each agent directory found will become a new 'iter'
            for agent_path in agent_paths:
                iter_num = iter_counter

                step_paths = sorted(
                    [d for d in agent_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
                    key=lambda p: numeric_suffix(p.name, "step_")
                )

                # If an agent directory has no steps, we skip creating an iter for it.
                if not step_paths:
                    continue

                # Create the destination directory for this new iter's attempts
                dst_attempts_root = dst_dir / f"task{task_num}" / "output" / "iter_output" / f"iter_{iter_num}" / "attempts"
                dst_attempts_root.mkdir(parents=True, exist_ok=True)

                # Each 'step' inside this agent becomes an 'attempt'
                for attempt_num, step_path in enumerate(step_paths):
                    dst_attempt_path = dst_attempts_root / f"attempt_{attempt_num}"
                    dst_attempt_path.mkdir(exist_ok=True)

                    # --- File Conversion ---

                    # a) Copy files
                    copy_file(step_path / "kernel.py", dst_attempt_path / "code.py")
                    copy_file(step_path / "prompt.md", dst_attempt_path / "prompt.md")
                    copy_file(step_path / "query_result.md", dst_attempt_path / "response.md")
                    copy_file(step_path / "query_result_full_llm_response.json", dst_attempt_path / "raw_response.json")

                    src_eval_file = step_path / "eval_results.json"
                    dst_metrics_file = dst_attempt_path / "metrics_artifacts.json"

                    # b) Transform eval_results.json to metrics_artifacts.json
                    metrics_data = {}
                    if src_eval_file.exists():
                        try:
                            with src_eval_file.open("r") as f:
                                eval_data = json.load(f)
                            metrics_data["artifacts"] = {
                                "results": eval_data,
                            }
                            runtime = eval_data.get("eval_results", {}).get("runtime")
                            if runtime is not None:
                                metrics_data["metrics"] = {"runtime": runtime}
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"    - Warning: Could not process {src_eval_file}. Error: {e}")
                    
                    with dst_metrics_file.open("w") as f:
                        json.dump(metrics_data, f, indent=4)

                    total_attempts_converted += 1

                # Crucially, increment the iter_counter AFTER processing all steps for this agent
                iter_counter += 1

        print(f"  - Converted {total_attempts_converted} total attempts into {iter_counter} iters for {task_name}.")

    print("\nConversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert a "prev" output directory structure to the "openevolve" structure.'
    )
    parser.add_argument("--src", type=str, required=True, help="Path to the source directory containing the run.")
    parser.add_argument("--dst", type=str, required=True, help="Path to the destination directory for the new structure.")
    args = parser.parse_args()

    # This path logic is specific to the "prev" structure's nesting.
    # It navigates down to the directory containing the 'task_*' folders.
    base_src_dir = Path(args.src)
    src_levels_dir = base_src_dir / "levels"
    level_name = os.listdir(src_levels_dir)[0]
    src_dir = src_levels_dir / level_name

    # The destination follows the OpenEvolve convention.
    dst_dir = Path(args.dst) / "tasks"

    os.makedirs(dst_dir, exist_ok=True)

    convert(src_dir, dst_dir)
